import os
import mshr
import dolfin
from block import block_mat, block_vec, block_bc
# from petsc4py import PETSc
from dolfin import *
from xii import *
import itertools
import hsmg

from biot_stokes_domains import *

# TODO: the below doesn't actually silence FFC warnings
# FFC complains when you try to set Dirichlet BCs on bubble elements, but it's probably fine :)
import logging
logging.getLogger('FFC').setLevel(logging.CRITICAL)

T = 1E-3
Nt = 10


class BiotStokesProblem(object):
    @staticmethod
    def default_params():
        return {
            "dt": 1.,
            "alpha": 1,
            "s0": 1.,
            "mu_f": 1.,
            "mu_p": 1,
            "lbd_p": 1.,
            "K": 1,
            "alpha_BJS": 1,
            "Cp": 0.
        }

    @staticmethod
    def brain_params():
        
        params = {
            "dt": 1E-3,

            # (Wirth B, Sobey I. 2009. Analytic solution during an infusion test of the linear unsteady poroelastic equations in a spherically symmetric model of the brain.)
            "alpha": 0.9995,

            # "s0": 1/lbd_p,

            # viscosity of water (Wikipedia)
            "mu_f": 0.8E-3,     # Pa*s

            # (Støverud, Darcis, Helmig, Hassanizadeh 2012)
            "mu_p": 1786,       # Pa
            "lbd_p": 7142,      # Pa

            # Holter, et al. 
            # "K": 1.9E-17,       # m^2,
            "K": 1.9E-5,       # micron^2,

            # ???
            "alpha_BJS": 1E-2,
            "Cp": 0
            
        }
        params["s0"] = 1 / float(params["lbd_p"])

        return pars
    
    def __init__(self, domain, param_dict, order=2):
        d = BiotStokesProblem.default_params()
        d.update(param_dict)
        self.params = d

        self.domain = domain

        self._dirichlet_bcs = {
            "stokes": {},
            "biot": {},
            "darcy": {}
        }
        self._neumann_bcs = {
            "stokes": {},
            "biot": {},
            "darcy": {}
        }

        self.make_function_spaces(order=order)

    def add_dirichlet_bc(self, problem_name, subdomain_id, value):
        bc_dict = self._dirichlet_bcs[problem_name]
        bc_dict[subdomain_id] = value

    def add_neumann_bc(self, problem_name, subdomain_id, value):
        bc_dict = self._neumann_bcs[problem_name]
        bc_dict[subdomain_id] = value

    def make_function_spaces(self, order=2):

        if order == 1:
            # biot
            Vp = FunctionSpace(self.domain.porous_domain, "RT", 1)
            Qp = FunctionSpace(self.domain.porous_domain, "DG", 0)
            U = VectorFunctionSpace(self.domain.porous_domain, "CG", 1)

            # stokes
            elt = triangle if self.domain.dimension == 2 else tetrahedron
            V = FiniteElement('Lagrange', elt, 1)
            Vb = FiniteElement(
                'Bubble', elt, 3 if self.domain.dimension == 2 else 4)
            mini = V + Vb
            mini_vec = VectorElement(mini, 2)
            # dolfin.set_log_level(30)          # mini element not implemented well

            # dolfin.set_log_level(dolfin.CRITICAL)
            Vf = FunctionSpace(self.domain.stokes_domain, mini_vec)
            Qf = FunctionSpace(self.domain.stokes_domain, "CG", 1)

            # lagrange multiplier
            X = FunctionSpace(self.domain.interface, "DG", 0)
        else:
            # biot
            Vp = FunctionSpace(self.domain.porous_domain, "RT", 2)
            Qp = FunctionSpace(self.domain.porous_domain, "DG", 1)
            U = VectorFunctionSpace(self.domain.porous_domain, "CG", 2)

            # stokes
            Vf = VectorFunctionSpace(self.domain.stokes_domain, "CG", 2)
            Qf = FunctionSpace(self.domain.stokes_domain, "CG", 1)

            # lagrange multiplier
            X = FunctionSpace(self.domain.interface, "DG", 1)

        self.W = [Vp, Qp, U, Vf, Qf, X]
        print("dofs: {}".format(sum([sp.dim() for sp in self.W])))

    def get_source_terms(self):
        """Override this to add source terms to the RHS
        (s_vp, s_wp, s_ep, s_vf, s_wf)"""
        D = self.domain.dimension
        return [
            Constant([0] * D),
            Constant(0),
            Constant([0] * D),
            Constant([0] * D),
            Constant(0)
        ]

    def get_initial_conditions(self):
        """Override this to add initial conditions different from zero.
        Order: up, pp, dp, uf, pp. (no lambda)"""
        D = self.domain.dimension
        return [
            Constant([0] * D),
            Constant(0),
            Constant([0] * D),
            Constant([0] * D),
            Constant(0),
        ]

    def get_solver(self):
        """Returns an iterator over solution values. Values are returned as a 
        list of Functions, with the ordering being [up, pp, dp, uf, pf, lbd]. 
        First returned value is initial conditions."""

        # names of params

        dt = self.params["dt"]
        alpha = self.params["alpha"]
        alpha_BJS = self.params["alpha_BJS"]
        s0 = self.params["s0"]
        mu_f = self.params["mu_f"]
        mu_p = self.params["mu_p"]
        lbd_p = self.params["lbd_p"]
        K = self.params["K"]
        Cp = self.params["Cp"]

        C_BJS = (mu_f * alpha_BJS) / sqrt(K)

        # names of things needed to build matrices
        dxGamma = Measure("dx", domain=self.domain.interface)
        dxDarcy = Measure("dx", domain=self.domain.porous_domain)
        dxStokes = Measure("dx", domain=self.domain.stokes_domain)

        dsDarcy = Measure("ds", domain=self.domain.porous_domain,
                          subdomain_data=self.domain.porous_bdy_markers)
        dsStokes = Measure("ds", domain=self.domain.stokes_domain,
                           subdomain_data=self.domain.stokes_bdy_markers)

        up, pp, dp, uf, pf, lbd = map(TrialFunction, self.W)
        vp, wp, ep, vf, wf, mu = map(TestFunction, self.W)
        up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev = map(
            Function, self.W)

        # thank you, Miro!
        Tdp, Tuf = map(lambda x: Trace(x, self.domain.interface),
                       [dp, uf]
                       )

        Tep, Tvf = map(lambda x: Trace(x, self.domain.interface),
                       [ep, vf]
                       )

        n_Gamma_f = self.domain.interface_normal_f
        n_Gamma_p = self.domain.interface_normal_p

        # last argument is a point in the interior of the
        #  domain the normal should point outwards from

        # should be removed when not in the MMS domain
        assert n_Gamma_f(Point(0.0, 0.0))[1] == -1
        assert n_Gamma_p(Point(0.0, 0.0))[1] == 1

        # tau = Constant(((0, -1),
        #                 (1, 0))) * n_Gamma_f

        Tup = Trace(up, self.domain.interface)
        Tvp = Trace(vp, self.domain.interface)

        # a bunch of forms
        af = Constant(2 * mu_f) * inner(sym(grad(uf)),
                                        sym(grad(vf))) * dxStokes

        mpp = pp * wp * dx

        adp = Constant(mu_p / K) * inner(up, vp) * dxDarcy
        aep = (
            Constant(2 * mu_p) * inner(sym(grad(dp)), sym(grad(ep))) * dxDarcy
            + Constant(lbd_p) * inner(div(dp), div(ep)) * dxDarcy
        )
        bpvp = - inner(div(vp), pp) * dxDarcy
        bpvpt = - inner(div(up), wp) * dxDarcy
        bpep = - inner(div(ep), pp) * dxDarcy
        bpept = - inner(div(dp), wp) * dxDarcy
        bf = - inner(div(vf), pf) * dxStokes
        bft = - inner(div(uf), wf) * dxStokes

        # matrices living on the interface
        npvp, npep, nfvf = [
            lbd * inner(testfunc, n) * dxGamma
            for (testfunc, n) in [(Tvp, n_Gamma_p), (Tep, n_Gamma_p), (Tvf, n_Gamma_f)]
        ]
        npvpt, npept, nfvft = [
            mu * inner(trialfunc, n) * dxGamma
            for (trialfunc, n) in [(Tup, n_Gamma_p), (Tdp, n_Gamma_p), (Tuf, n_Gamma_f)]
        ]

        svfuf, svfdp, sepuf, sepdp = [
            Constant(C_BJS) * (
                inner(testfunc, trialfunc) * dxGamma
                - inner(testfunc, n_Gamma_f) *
                inner(trialfunc, n_Gamma_f) * dxGamma
            )
            # Constant(C_BJS) * (
            #     inner(testfunc, tau) * inner(trialfunc, tau) * dxGamma
            # )

            for (testfunc, trialfunc) in [
                (Tvf, Tuf), (Tvf, Tdp), (Tep, Tuf), (Tep, Tdp)
            ]
        ]

        Co = Constant
        a = [
            [adp, bpvp, 0, 0, 0, npvp],
            [bpvpt, -Co(s0 / dt) * mpp, Co(alpha / dt) * bpept, 0, 0, 0],
            [0, Constant(alpha / dt) * bpep, Co(1 / dt) * (aep + Co(1 / dt) * sepdp),
             -Co(1 / dt) * sepuf, 0, Co(1 / dt) * npep],
            [0, 0, Co(-1 / dt) * svfdp, af + svfuf, bf, nfvf],
            [0, 0, 0, bft, 0, 0],
            [npvpt, 0, Co(1 / dt) * npept, nfvft, 0, 0],
        ]

        def compute_RHS(dp_prev, pp_prev, neumann_bcs, t):
            nf = FacetNormal(self.domain.stokes_domain)
            np = FacetNormal(self.domain.porous_domain)

            Tdp_prev = Trace(dp_prev, self.domain.interface)
            s_vp, s_wp, s_ep, s_vf, s_wf = self.get_source_terms()

            # update t in source terms
            for expr in (s_vp, s_wp, s_ep, s_vf, s_wf):
                expr.t = t

            # update t in neumann bcs
            for prob_name in ["biot", "darcy", "stokes"]:
                for expr in neumann_bcs[prob_name].values():
                    expr.t = t

            biot_neumann_terms = sum(  # val = sigma_p
                (inner(dot(val, np), ep) * dsDarcy(subdomain)
                 for subdomain, val in neumann_bcs["biot"].items())
            )
            stokes_neumann_terms = sum(  # val = sigma_f
                (inner(dot(val, nf), vf) * dsStokes(subdomain)
                 for subdomain, val in neumann_bcs["stokes"].items())
            )
            darcy_neumann_terms = sum(  # val = -pp
                (dot(val * np, vp) * dsDarcy(subdomain)
                 for subdomain, val in neumann_bcs["darcy"].items())
            )

            bpp = (
                Constant(s0 / dt) * pp_prev * wp * dxDarcy
                + Constant(alpha / dt) * inner(div(dp_prev), wp) * dxDarcy
            )

            L_Cp_vp = Constant(Cp) * inner(Tvp, n_Gamma_p) * dxGamma
            L_Cp_ep = Constant(Cp) * inner(Tep, n_Gamma_p) * dxGamma

            L_BJS_vf = -Constant(C_BJS / dt) * (
                inner(Tdp_prev, Tvf) * dxGamma
                - inner(Tdp_prev, n_Gamma_f) * inner(Tvf, n_Gamma_f) * dxGamma
                # inner(Tdp_prev, tau) * inner(Tvf, tau) * dxGamma
            )
            L_BJS_ep = Constant(C_BJS / dt) * (
                inner(Tdp_prev, Tep) * dxGamma
                - inner(Tdp_prev, n_Gamma_f) * inner(Tep, n_Gamma_f) * dxGamma
                # inner(Tdp_prev, tau) * inner(Tep, tau) * dxGamma
            )

            L_mult = Constant(1 / dt) * inner(
                Tdp_prev, n_Gamma_p
            ) * mu * dxGamma

            # source terms
            S_vp = inner(s_vp, vp) * dxDarcy
            S_wp = inner(s_wp, wp) * dxDarcy
            S_ep = inner(s_ep, ep) * dxDarcy
            S_vf = inner(s_vf, vf) * dxStokes
            S_wf = inner(s_wf, wf) * dxStokes

            L = [
                L_Cp_vp + Constant(mu_p / K) * S_vp + darcy_neumann_terms,
                -(bpp + S_wp),
                Co(1 / dt) * (L_Cp_ep + S_ep + L_BJS_ep + biot_neumann_terms),
                S_vf + L_BJS_vf + stokes_neumann_terms,
                -S_wf,
                L_mult

            ]
            return L

        up_bcs = [
            DirichletBC(
                self.W[0], self._dirichlet_bcs["darcy"][subdomain_num],
                self.domain.porous_bdy_markers, subdomain_num
            )
            for subdomain_num in self._dirichlet_bcs["darcy"]
        ]

        dp_bcs = [
            DirichletBC(
                self.W[2], self._dirichlet_bcs["biot"][subdomain_num],
                self.domain.porous_bdy_markers, subdomain_num
            )
            for subdomain_num in self._dirichlet_bcs["biot"]
        ]

        uf_bcs = [
            DirichletBC(
                self.W[3], self._dirichlet_bcs["stokes"][subdomain_num],
                self.domain.stokes_bdy_markers, subdomain_num
            )
            for subdomain_num in self._dirichlet_bcs["stokes"]
        ]

        bcs = [
            up_bcs,
            [],                 # pp
            dp_bcs,
            uf_bcs,
            [],                 # pf
            []                  # lbd
        ]

        AA = ii_assemble(a)

        def update_t_in_dirichlet_bcs(t):
            all_bc_exprs = (
                self._dirichlet_bcs["darcy"].values()
                + self._dirichlet_bcs["biot"].values()
                + self._dirichlet_bcs["stokes"].values()
            )
            for expr in all_bc_exprs:
                expr.t = t

        bbcs = block_bc(bcs, symmetric=True)
        AA = ii_convert(AA, "")
        AA = set_lg_map(AA)
        bbcs = bbcs.apply(
            AA
        )
        AAm = ii_convert(AA)
        # assert np.all(AAm.array() == AAm.array())

        w = ii_Function(self.W)

        solver = LUSolver('default')
        solver.set_operator(AAm)
        solver.parameters['reuse_factorization'] = True

        # alright, we're all set - now set initial conditions and solve
        t = 0

        initial_conditions = self.get_initial_conditions()
        for func, func_0 in zip(
                [up_prev, pp_prev, dp_prev, uf_prev, pf_prev], initial_conditions
        ):
            func_0.t = 0
            func.assign(func_0)

        yield t, [up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev]

        while True:
            # solve for next time step
            t += dt
            update_t_in_dirichlet_bcs(t)

            L = compute_RHS(dp_prev, pp_prev, self._neumann_bcs, t)
            bb = ii_assemble(L)
            bbcs.apply(bb)
            bbm = ii_convert(bb)

            solver.solve(w.vector(), bbm)

            for i, func in enumerate([up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev]):
                func.assign(w[i])

            yield t, w


class AmbartsumyanMMSProblem(BiotStokesProblem):
    def __init__(self, N, user_params, order):
        domain = AmbartsumyanMMSDomain(N)
        params = BiotStokesProblem.default_params()

        for k in user_params:
            assert k in params

        if "Cp" in user_params or "dt" in user_params:
            raise Exception()   # todo: fix
            
        params.update(user_params)
        
        
        params["dt"] = T / Nt

        super(AmbartsumyanMMSProblem, self).__init__(
            domain, params, order=order)
        up_e, _, dp_e, _, _ = self.exact_solution()

        for prob_name, exact_sol in zip(
                ["darcy", "biot"],
                [up_e, dp_e]
        ):
            self.add_dirichlet_bc(prob_name, 2, exact_sol)

        stokes_neumann = Expression(
            (
                ("-8*pi*mu_f*cos(pi*t) - exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])",                        "-pi*mu_f*sin(x[1])*cos(pi*t)"                     ),
                (                      "-pi*mu_f*sin(x[1])*cos(pi*t)"                     ,               "-exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])"             )
            ), degree=6, t=0, **self.params
        )

        self.add_neumann_bc("stokes", 2, stokes_neumann)
        
        


    def get_initial_conditions(self):
        exprs = self.exact_solution()
        for expr in exprs:
            expr.t = 0
        return exprs

    def exact_solution(self):
        up_e=Expression(
            (
                "pi*exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1])",
                "(1.0/2.0)*pi*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
            ), degree=5, t=0, **self.params
        )
        pp_e=Expression(
            "exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])", degree=5, t=0, **self.params
        )
        dp_e=Expression(
            (
                "-(3*x[0] - cos(x[1]))*sin(pi*t) + 1",
                "(lbd_p + 2*mu_p + (3*lbd_p*x[1] + lbd_p + 2*mu_p)*sin(pi*t))/(lbd_p + 2*mu_p)"
            ), degree=5, t=0, **self.params
        )
        uf_e=Expression(
            (
                "pi*(-3*x[0] + cos(x[1]))*cos(pi*t)",
                "pi*(x[1] + 1)*cos(pi*t)"
            ), degree=5, t=0, **self.params
        )
        pf_e=Expression(
            "2*pi*mu_f*cos(pi*t) + exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])", degree=5, t=0, **self.params
        )

        return up_e, pp_e, dp_e, uf_e, pf_e

    def get_source_terms(self):
        s_vp = Expression(
            (
                "pi*(K + mu_p)*exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1])/mu_p",
                "(1.0/2.0)*pi*(-K + mu_p)*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])/mu_p"
            ), degree=6, t=0, **self.params
        )
        s_wp = Expression(
            "alpha*(3*pi*lbd_p*cos(pi*t)/(lbd_p + 2*mu_p) - 3*pi*cos(pi*t)) + s0*exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1]) - 3.0/4.0*pow(pi, 2)*exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])", degree=6, t=0, **self.params
        )
        s_ep =  Expression(
            (
                "pi*alpha*exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1]) + mu_p*sin(pi*t)*cos(x[1])",
                "-1.0/2.0*pi*alpha*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
            ), degree=6, t=0, **self.params
        )
        s_vf = Expression(
            (
                "pi*(mu_f*cos(x[1])*cos(pi*t) + exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1]))",
                "-1.0/2.0*pi*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
            ), degree=6, t=0, **self.params
        )
        s_wf =  Expression(
            "-2*pi*cos(pi*t)", degree=6, t=0, **self.params
        )

        return s_vp, s_wp, s_ep, s_vf, s_wf

    def compute_errors(self, funcs, t, norm_types):
        """Given a list of solution values, set the time in all the exact sol
        expressions and return a list of the errornorms."""

        exprs = self.exact_solution()
        for e in exprs:
            e.t = t

        return [
            errornorm(
                expr, func,
                norm_type=norm, degree_rise=3,
                mesh=func.function_space().mesh())
            for func, expr, norm in zip(funcs, exprs, norm_types)
        ]

    def save_exact_solution_to_file(self, t, fs):
        exprs = self.exact_solution()
        W = self.W

        names = ["up", "pp", "dp", "uf", "pf"]
        for i, (expr, function_space, f, name) in enumerate(
                zip(exprs, list(W)[:-1], fs, names)
        ):
            expr.t = t
            u = interpolate(expr, function_space)
            u.rename(name, str(i))
            f << u


class TunnelProblem(BiotStokesProblem):
    def __init__(self, params, N, pi, order=2):
        domain = Tunnel2DDomain(N)
        params = BiotStokesProblem.default_params()

        

        super(AmbartsumyanMMSProblem, self).__init__(
            domain, params, order=order)


        # dirichlet BCs on walls
        zero_vec = Constant((0, 0))
        self.add_dirichlet_bc("darcy", 2, zero_vec)
        

        # # neumann terms
        # darcy_neumann = Expression(
        #     "-exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])", degree=5, t=0
        # )
        # biot_neumann = Expression(
        #     (
        #         ("-exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1]) - 8*sin(pi*t)", "-sin(x[1])*sin(pi*t)"),
        #         ("-sin(x[1])*sin(pi*t)","-exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])")
        #     ), degree=5, t=0
        # )
        stokes_neumann = Expression(
            (
                ("p", "0"),
                ("0", "p")
            ), degree=5, t=0, p=pi
        )

        # self.add_neumann_bc("darcy", 2, darcy_neumann)
        # self.add_neumann_bc("biot", 2, biot_neumann)
        self.add_neumann_bc("stokes", 2, stokes_neumann)


def save_to_file(things, fs, names):
    for i, (thing, f, name) in enumerate(zip(things, fs, names)):
        thing.rename(name, str(i))
        f << thing


names = ["up", "pp", "dp", "uf", "pf", "lbd"]
normlist = ["L2", "L2", "H1", "H1", "L2"]
result_dir = "mms_results"


def in_dir(fn):
    return os.path.join(result_dir, fn)


solution_fs = map(File, map(in_dir, ["up.pvd", "pp.pvd",
                                     "dp.pvd", "uf.pvd", "pf.pvd", "lbd.pvd"]))
exact_fs = map(File, map(in_dir, ["up_e.pvd", "pp_e.pvd",
                                  "dp_e.pvd", "uf_e.pvd", "pf_e.pvd"]))


# now solve
N0 = 10
print "Using {} timesteps and T={}".format(Nt, T)

errs = {}

import random



for N in [N0, 2 * N0]:

    mms_params = {
        "alpha": 0.9995,
        "mu_f": 0.8E-3,
        "lbd_p": 7142.,        # gets bad if bumped up an order of magnitude
        "mu_p": 1786,
        "K": 1E-5,                 # gets bad at 1E-1, needs to be 1E-5
        "alpha_BJS": 1E-2,      # gets bad at 1E-3
    }

    mms_params["s0"] = 1/mms_params["lbd_p"]


    problem = AmbartsumyanMMSProblem(N, mms_params, order=2)
    solution = problem.get_solver()

    # Nt = 10
    for i in range(Nt + 1):
        t, funcs = solution.next()
        print "\rDone with timestep {:>3d} of {}".format(i, Nt),
        save_to_file(funcs, solution_fs, names)
        problem.save_exact_solution_to_file(t, exact_fs)

    # print errors at final time step
    print "\n"
    # print "N={}:".format(N)
    print "Done with N={}".format(N)
    err_N = {}
    for func, expr, name, normtype in zip(
            list(funcs)[:5], problem.exact_solution(),
            names[:5], normlist
    ):
        expr.t = t
        error = errornorm(
            expr, func,
            norm_type=normtype, degree_rise=3,
            mesh=func.function_space().mesh()
        )
        err_N[name] = error
        s = (
            "err_{}: {:.10f} (norm={})".format(
                name, error, normtype
            )
        )
        # print s
    errs[N] = err_N


print "convergence rate going from N={} to N={}:".format(N0, 2 * N0)
for name in names[:5]:
    import math
    k = math.log(errs[N0][name] / errs[2 * N0][name]) / math.log(2)
    print "{}: {:.5f}".format(name, k)
