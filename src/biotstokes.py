import mshr
import dolfin
from block import block_mat, block_vec, block_bc
# from petsc4py import PETSc
from dolfin import *
from xii import *
import itertools
import hsmg


class AmbartsumyanMMSDomain(object):
    def __init__(self, N):

        _mesh = RectangleMesh(Point(0, -1), Point(1, 1), N, 2 * N)

        stokes_subdomain = dolfin.CompiledSubDomain("x[1] > 0")

        subdomains = MeshFunction('size_t', _mesh, _mesh.topology().dim(), 0)
        # Awkward marking
        for cell in cells(_mesh):
            x = cell.midpoint().array()
            subdomains[cell] = int(stokes_subdomain.inside(x, False))

        self.full_domain = _mesh
        self.stokes_domain = EmbeddedMesh(subdomains, 1)
        self.porous_domain = EmbeddedMesh(subdomains, 0)

        surfaces = MeshFunction('size_t', self.porous_domain,
                                self.porous_domain.topology().dim() - 1, 0)
        CompiledSubDomain("on_boundary && near(x[1], 0)").mark(surfaces, 1)
        self.interface = EmbeddedMesh(surfaces, 1)

        self.mark_boundary()

    @property
    def dimension(self):
        return 2

    def mark_boundary(self):
        """Interface should be marked as 0. Do not set BCs there.
        Other bdy is 1"""

        stokes_markers = FacetFunction("size_t", self.stokes_domain, 0)
        porous_markers = FacetFunction("size_t", self.porous_domain, 0)

        interface_bdy = dolfin.CompiledSubDomain(
            "near(x[1], 0) && on_boundary")
        other_bdy = dolfin.CompiledSubDomain("on_boundary")

        for markers in [stokes_markers, porous_markers]:
            other_bdy.mark(markers, 1)
            interface_bdy.mark(markers, 0)

        self.stokes_bdy_markers = stokes_markers
        self.porous_bdy_markers = porous_markers


class BiotStokesProblem(object):
    @staticmethod
    def default_params():
        return {
            "dt": 1.,
            "alpha": 1.,
            "s0": 1.,
            "mu_f": 1.,
            "mu_p": 1.,
            "lbd_f": 1.,
            "lbd_p": 1.,
            "K": 1.,
            "alpha_BJS": 1.,
            "Cp": 1.
        }

    def __init__(self, domain, param_dict):
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

        self.make_function_spaces()

    def add_dirichlet_bc(self, problem_name, subdomain_id, value):
        bc_dict = self._dirichlet_bcs[problem_name]
        bc_dict[subdomain_id] = value

    def add_neumann_bc(self, problem_name, subdomain_id, value):
        bc_dict = self._neumann_bcs[problem_name]
        bc_dict[subdomain_id] = value

    def make_function_spaces(self):
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
        (f_p, q_p, f_f, q_f)"""
        D = self.domain.dimension
        return [
            Constant([0] * D),
            Constant(0),
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
        First returned value is initial conditions (zero)."""

        # names of params

        dt = self.params["dt"]
        alpha = self.params["alpha"]
        alpha_BJS = self.params["alpha_BJS"]
        s0 = self.params["s0"]
        mu_f = self.params["mu_f"]
        mu_p = self.params["mu_p"]
        lbd_f = self.params["lbd_f"]
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


        # last argument is a point in the interior of the
        #  domain the normal should point outwards from
        n_Gamma_f = OuterNormal(self.domain.interface,
                                [0.5] * self.domain.dimension)
        # should be removed when not in the MMS domain
        assert n_Gamma_f(Point(0.0, 0.0))[1] == -1
        
        n_Gamma_p = OuterNormal(self.domain.interface,
                                [-0.5] * self.domain.dimension)
        assert n_Gamma_p(Point(0.0, 0.0))[1] == 1

        Tup = Trace(up, self.domain.interface, restriction="-", normal=n_Gamma_f)
        Tvp = Trace(vp, self.domain.interface, restriction="-", normal=n_Gamma_f)


        # a bunch of forms
        af =  Constant(2 * mu_f) * inner(sym(grad(uf)), sym(grad(vf))) * dxStokes
        
        mpp = pp * wp * dx
        
        adp = Constant(mu_p / K) * inner(up, vp) * dxDarcy
        aep = (
            Constant(mu_p) * inner(sym(grad(dp)), sym(grad(ep))) * dxDarcy
            + Constant(lbd_p) * inner(div(dp), div(ep)) * dxDarcy
        )
        bpvp  = - inner(div(vp), pp) * dxDarcy
        bpvpt = - inner(div(up), wp) * dxDarcy
        bpep  = - inner(div(ep), pp) * dxDarcy
        bpept = - inner(div(dp), wp) * dxDarcy
        bf    = - inner(div(vf), pf) * dxStokes
        bft   = - inner(div(uf), wf) * dxStokes

        # matrices living on the interface
        npvp, npep, nfvf = [
            lbd * inner(testfunc, n) * dxGamma
            for (testfunc, n) in [(Tvp, n_Gamma_p), (Tep, n_Gamma_p), (Tvf, n_Gamma_f)]
        ]
        npvpt, npept, nfvft = [
            mu * inner(trialfunc, n) * dxGamma
            for (trialfunc, n) in [(Tup, n_Gamma_p), (Tdp, n_Gamma_p), (Tuf, n_Gamma_f)]
        ]

        
        # to build sum_j ((a*tau_j), (b*tau_j)) we use a trick - see Thoughts
        svfuf, svfdp, sepuf, sepdp = [
            Constant(C_BJS) * (
                inner(testfunc, trialfunc)  * dxGamma
                - inner(testfunc, n_Gamma_f) * inner(trialfunc, n_Gamma_f) * dxGamma
            )
            for (testfunc, trialfunc) in [
                (Tvf, Tuf), (Tvf, Tdp), (Tep, Tuf), (Tep, Tdp)
            ]
        ]


        Co = Constant
        a = [
            [adp, bpvp, 0, 0, 0, npvp],
            [-bpvpt, Co(s0/dt)*mpp, Co(-alpha/dt)*bpept, 0, 0, 0],
            [0, Constant(alpha)*bpep, (aep + Co(1/dt)*sepdp), -sepuf, 0, npep],
            [0, 0, Co(-1/dt)*svfdp, af + svfuf, bf, nfvf],
            [0, 0, 0, -bft, 0, 0],
            [npvpt, 0, Co(1/dt)*npept, nfvft, 0, 0],
        ]

        

        
        
        # quick sanity check
        N_unknowns = 6
        assert len(a) == N_unknowns
        for row in a:
            assert len(row) == N_unknowns


        def compute_RHS(dp_prev, pp_prev, neumann_bcs, t):
            nf = FacetNormal(self.domain.stokes_domain)
            np = FacetNormal(self.domain.porous_domain)

            for prob_name in ["biot", "darcy", "stokes"]:  # update t in neumann bcs
                for expr in neumann_bcs[prob_name].keys():
                    expr.t = t

            biot_neumann_terms, darcy_neumann_terms, stokes_neumann_terms = (
                sum(
                    [
                        inner(
                            testfunc, neumann_bcs[prob_name][subdomain]
                        ) * measure(subdomain)
                        for subdomain in neumann_bcs[prob_name]
                    ]
                ) for prob_name, testfunc, measure in zip(
                    ["biot", "darcy", "stokes"],
                    [ep, vp, vf],
                    [dsDarcy, dsDarcy, dsStokes]
                )
            )

            s_vp, s_wp, s_ep, s_vf, s_wf = self.get_source_terms()

            Tdp_prev = Trace(dp_prev, self.domain.interface)            
            for expr in (
                    s_vp, s_wp, s_ep, s_vf, s_wf,
                    # g_up, g_pp, g_dp, g_uf, g_pf,
            ):
                expr.t = t

            L_darcy = (
                inner(vp, np) * Constant(0) * dsDarcy
                + darcy_neumann_terms
                
            )

            L_biot = (
                Constant(0) * inner(np, ep) * dsDarcy
                + biot_neumann_terms
            )
            L_stokes = (
                Constant(0) * inner(nf, vf) * dsStokes
                + stokes_neumann_terms
            )

            bpp = (
                Constant(s0 / dt) * pp_prev * wp * dxDarcy
                + Constant(alpha / dt) * inner(div(dp_prev), wp) * dxDarcy
            )

            L_Cp_vp =  Constant(Cp) * inner(Tvp, n_Gamma_p) * dxGamma
            L_Cp_ep =  Constant(Cp) * inner(Tep, n_Gamma_f) * dxGamma
            
            
            L_BJS_vf = -Constant(C_BJS)*(
                inner(Tdp_prev, Tvf)  * dxGamma
                - inner(Tdp_prev, n_Gamma_f) * inner(Tvf, n_Gamma_f) * dxGamma
            )
            L_BJS_ep = Constant(C_BJS)*(
                inner(Tdp_prev, -Tep)  * dxGamma
                - inner(Tdp_prev, n_Gamma_f) * inner(-Tep, n_Gamma_f) * dxGamma
            )
            

            L_mult = Constant(1 / dt) * inner(Tdp_prev,
                                              n_Gamma_p) * mu * dxGamma

            S_vp = inner(s_vp, vp) * dxDarcy
            S_wp = inner(s_wp, wp) * dxDarcy
            S_ep = inner(s_ep, ep) * dxDarcy
            S_vf = inner(s_vf, vf) * dxStokes
            S_wf = inner(s_wf, wf) * dxStokes
            

            L = [
                L_darcy + L_Cp_vp + S_vp,
                (bpp + Constant(mu_p/K)*S_wp),
                (
                    L_biot + L_Cp_ep + S_ep
                    + L_BJS_ep # this should be here
                ),
                (
                    L_stokes + S_vf + 
                    L_BJS_vf # this should be here
                )
                ,
                S_wf,
                L_mult

            ]

            assert len(L) == N_unknowns

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
        self.form = a
        AA = ii_assemble(a)

        def update_t_in_dirichlet_bcs(t):
            all_bc_exprs = (
                self._dirichlet_bcs["darcy"].values()
                + self._dirichlet_bcs["biot"].values()
                + self._dirichlet_bcs["stokes"].values()
            )
            for exp in all_bc_exprs:
                exp.t = t

        bbcs = block_bc(bcs, symmetric=True)
        AA = ii_convert(AA, "")
        AA = set_lg_map(AA)
        bbcs = bbcs.apply(
            AA
        )
        AAm = ii_convert(AA)

        self.matrix = AAm
        
        w = ii_Function(self.W)

        solver = LUSolver('default')  # this should be umfpack
        solver.set_operator(AAm)
        solver.parameters['reuse_factorization'] = True

        # alright, we're all set - now set initial conditions and solve
        t = 0

        initial_conditions = self.get_initial_conditions()
        for func, func_0 in zip(
                [up_prev, pp_prev, dp_prev, uf_prev, pf_prev], initial_conditions
        ):
            func_0.t = 0        # should be unnecessary
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
    def __init__(self, N):
        domain = AmbartsumyanMMSDomain(N)
        params = BiotStokesProblem.default_params()
        for k in params:
            params[k] = 1

        params["dt"] = 1E-4     # note: if you change this, mms_rhs will discretize wrongly
        params["alpha_BJS"] = 1
        params["Cp"] = 0


        super(AmbartsumyanMMSProblem, self).__init__(domain, params)
        up_e, _, dp_e, uf_e, _ = self.exact_solution()

        for prob_name, exact_sol in zip(
                ["darcy", "biot", "stokes"],
                [up_e, dp_e, uf_e]
        ):
            self.add_dirichlet_bc(prob_name, 1, exact_sol)



    def get_initial_conditions(self):
        exprs = self.exact_solution()
        for expr in exprs:
            expr.t = 0
        return exprs

    
    def exact_solution(self):

        up_e=Expression(
        (
          "-pi*exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1])",
          "(1.0/2.0)*pi*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
        ), degree=5, t=0
        )
        pp_e=Expression(
        "exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])", degree=5, t=0
        )
        dp_e=Expression(
        (
          "(-3*x[0] + cos(x[1]))*sin(pi*t)",
          "(x[1] + 1)*sin(pi*t)"
        ), degree=5, t=0
        )
        uf_e=Expression(
        (
          "pi*(-3*x[0] + cos(x[1]))*cos(pi*t)",
          "pi*(x[1] + 1)*cos(pi*t)"
        ), degree=5, t=0
        )
        pf_e=Expression(
        "exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1]) + 2*pi*cos(pi*t)", degree=5, t=0
        )

        return up_e, pp_e, dp_e, uf_e, pf_e

    def get_source_terms(self):

        s_vp = Expression(
        (
          "0",
          "0"
        ), degree=5, t=0
        )
        s_wp = Expression(
        "exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1]) + (5.0/4.0)*pow(pi, 2)*exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1]) - 2*pi*cos(pi*t)", degree=5, t=0
        )
        s_ep =  Expression(
        (
          "pi*exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1]) + sin(pi*t)*cos(x[1])",
          "-1.0/2.0*pi*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
        ), degree=5, t=0
        )
        s_vf = Expression(
        (
          "pi*(exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1]) + cos(x[1])*cos(pi*t))",
          "-1.0/2.0*pi*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
        ), degree=5, t=0
        )
        s_wf =  Expression(
        "-2*pi*cos(pi*t)", degree=5, t=0
        )

        return s_vp, s_wp, s_ep, s_vf, s_wf

    # def exact_solution(self):
    #     up_e=Expression(
    #     (
    #       "pow(x[0], 2)*x[1]",
    #       "x[0]"
    #     ), degree=5, t=0
    #     )
    #     pp_e=Expression(
    #     "2*x[0] + pow(x[1], 3)", degree=5, t=0
    #     )
    #     dp_e=Expression(
    #     (
    #       "x[0] - x[1]",
    #       "x[0] + x[1]"
    #     ), degree=5, t=0
    #     )
    #     uf_e=Expression(
    #     (
    #       "2*x[1]",
    #       "pow(x[0], 4)"
    #     ), degree=5, t=0
    #     )
    #     pf_e=Expression(
    #     "2*x[0] - 1", degree=5, t=0
    #     )

    #     return up_e, pp_e, dp_e, uf_e, pf_e

    # def get_source_terms(self):
    #     s_uf = Expression(
    #     (
    #       "2",
    #       "-12*pow(x[0], 2)"
    #     ), degree=5, t=0
    #     )
    #     s_pf =  Expression(
    #     "0", degree=5, t=0
    #     )
    #     s_up = Expression(
    #     (
    #       "pow(x[0], 2)*x[1] + 2",
    #       "x[0] + 3*pow(x[1], 2)"
    #     ), degree=5, t=0
    #     )
    #     s_dp =  Expression(
    #     (
    #       "2",
    #       "3*pow(x[1], 2)"
    #     ), degree=5, t=0
    #     )
    #     s_pp = Expression(
    #     "2*x[0]*x[1]", degree=5, t=0
    #     )

    #     return s_up, s_pp, s_dp, s_uf, s_pf
 

    def compute_errors(self, funcs, t, norm_types=None):
        """Given a list of solution values, set the time in all the exact sol
        expressions and return a list of the errornorms."""

        if norm_types == None:
            norm_types = ["l2"]*5

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

        for i, (expr, function_space, f) in enumerate(zip(exprs, list(W)[:-1], fs)):
            expr.t = t
            u = interpolate(expr, function_space)
            u.rename("u", str(i))
            f << u
            



def save_to_file(things, fs):
    for i, (thing, f) in enumerate(zip(things, fs)):
        thing.rename("u", str(i))
        f << thing

def save_errors(t, funcs, exprs, fns, norm_types):
    """For each (f, e) pair, computes the errornorm and apepnds it to a file."""
    N = len(funcs)
    for l in (exprs, fns, norm_types):
        assert len(l) == N

    errs = problem.compute_errors(list(funcs)[:5], t)
    for fn, err in zip(fns, errs):
        with open(fn, "a") as f:
            f.write(
                "{:.4f}: {:.10f}\n".format(
                    t, err
                )
            )

N = 40
problem = AmbartsumyanMMSProblem(N)

solution = problem.get_solver()



names = ["up", "pp", "dp", "uf", "pf", "lbd"]
# global funcs
result_dir = "mms_results"

def in_dir(fn):
    return os.path.join(result_dir, fn)

solution_fs = map(File, map(in_dir, ["up.pvd", "pp.pvd",
                                     "dp.pvd", "uf.pvd", "pf.pvd", "lbd.pvd"]))
exact_fs = map(File, map(in_dir, ["up_e.pvd", "pp_e.pvd",
                                   "dp_e.pvd", "uf_e.pvd", "pf_e.pvd"]))
err_fns = map(in_dir, ["up_e.txt", "pp_e.txt",
                       "dp_e.txt", "uf_e.txt", "pf_e.txt"])
# # cleanup old files
# import itertools
for fn in err_fns:
    try:
        os.remove(fn)
    except OSError:
        pass

Nt = 1
for i in range(Nt + 1):
    t, funcs = solution.next()
    print "\r Done with timestep {:>3d} of {}".format(i, Nt),
    save_to_file(funcs, solution_fs)
    problem.save_exact_solution_to_file(t, exact_fs)
    
    # save_to_file(exprs, exact_solution_fns)
    save_errors(t, list(funcs)[:5], problem.exact_solution(),
                err_fns, ["l2"] * 5)

print ""
for func, expr, name in zip(list(funcs)[:5], problem.exact_solution(), names[:5]):
    # func2 = Function(func.function_space())
    # func2.assign(expr)
    expr.t = t
    s = (
        "err_{}: {:.10f}\n".format(
            name, errornorm(
                expr, func,
                norm_type="l2", degree_rise=3,
                mesh=func.function_space().mesh()
            )
        )
    )
    print s

