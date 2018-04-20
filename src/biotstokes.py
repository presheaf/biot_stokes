import mshr
import dolfin
from block import block_mat, block_vec, block_bc
# from petsc4py import PETSc
from dolfin import *
from xii import *
import itertools

class CylinderBoxDomain3D(object):
    def __init__(self, hi, ho):
        EPS = 1E-3
        R = 0.25
        # box_domain = mshr.Box(dolfin.Point(0, 0, 0), dolfin.Point(1, 1, 1))
        # _mesh = mshr.generate_mesh(box_domain, N)
        _mesh = make_cylinder_box_mesh("mymesh.xml", hi, ho, R)

        stokes_subdomain = dolfin.CompiledSubDomain(
            "sqrt(x[0] * x[0] + x[1] * x[1]) < R", R=R
        )

        subdomains = MeshFunction('size_t', _mesh, _mesh.topology().dim(), 0)

        # Awkward marking
        for cell in cells(_mesh):
            x = cell.midpoint().array()
            if stokes_subdomain.inside(x, False):
                subdomains[cell] = 1
            else:
                subdomains[cell] = 0

        submeshes, interface, _ = mortar_meshes(
            subdomains, range(2), strict=True, tol=EPS
        )

        self.full_domain = _mesh
        self.stokes_domain = submeshes[1]
        self.porous_domain = submeshes[0]
        self.interface = interface

        self.mark_boundary()

    @property
    def dimension(self):
        return 3

    def mark_boundary(self):
        """Interface should be marked as 0. Do not set BCs there.
        left, right, top, bottom = 1, 2, 3, 4"""

        stokes_markers = FacetFunction("size_t", self.stokes_domain, 0)
        porous_markers = FacetFunction("size_t", self.porous_domain, 0)

        interface_bdy = dolfin.CompiledSubDomain("on_boundary")

        left_bdy = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
        right_bdy = dolfin.CompiledSubDomain("near(x[0], 1) && on_boundary")
        top_bdy = dolfin.CompiledSubDomain("near(x[1], 0) && on_boundary")
        bottom_bdy = dolfin.CompiledSubDomain("near(x[1], 1) && on_boundary")

        for markers in [stokes_markers, porous_markers]:
            interface_bdy.mark(markers, 0)
            left_bdy.mark(markers, 1)
            right_bdy.mark(markers, 2)
            top_bdy.mark(markers, 3)
            bottom_bdy.mark(markers, 4)

        self.stokes_bdy_markers = stokes_markers
        self.porous_bdy_markers = porous_markers


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
            "near(x[1], 1) && on_boundary")
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
            "dt": 0.1,
            "alpha": 1.,
            "alpha_BJS": 1.,
            "s0": 1.,
            "mu_f": 1.,
            "mu_p": 1.,
            "lbd_f": 1.,
            "lbd_p": 1.,
            "K": 1.,
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
        Vp = FunctionSpace(self.domain.porous_domain, "RT", 1)
        Qp = FunctionSpace(self.domain.porous_domain, "DG", 0)
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
        Tup, Tdp, Tuf = map(lambda x: Trace(x, self.domain.interface),
                            [up, dp, uf]
                            )

        Tvp, Tep, Tvf = map(lambda x: Trace(x, self.domain.interface),
                            [vp, ep, vf]
                            )

        # todo: verify that this is actually the right normal
        n_Gamma_f = OuterNormal(self.domain.interface, [
                                0.5] * self.domain.dimension)
        n_Gamma_p = -n_Gamma_f

        # build a bunch of matrices

        mpp = Constant(s0 / dt) * pp * wp * dx

        adp = Constant(mu_p / K) * inner(up, vp) * dxDarcy
        aep = Constant(1/dt) * (
            Constant(mu_p) * inner(sym(grad(dp)), sym(grad(ep))) * dx
            + Constant(lbd_p) * inner(div(dp), div(ep)) * dx
        )
        af = Constant(2 * mu_f) * inner(sym(grad(uf)), sym(grad(vf))) * dx

        bpvp = - inner(div(vp), pp) * dxDarcy
        bpvpt = - inner(div(up), wp) * dx
        bpep = - Constant(alpha / dt) * inner(div(ep), pp) * dx
        bpept = - Constant(alpha / dt) * inner(div(dp), wp) * dx
        bf = - inner(div(vf), pf) * dx
        bft = - inner(div(uf), wf) * dx

        # matrices living on the interface
        npvp, npep, nfvf = [
            lbd * dot(testfunc, n) * dxGamma
            for (testfunc, n) in [(Tvp, n_Gamma_p), (Tep, n_Gamma_p), (Tvf, n_Gamma_f)]
        ]
        npvpt, npept, nfvft = [
            mu * dot(trialfunc, n) * dxGamma
            for (trialfunc, n) in [(Tup, n_Gamma_p), (Tdp, n_Gamma_p), (Tuf, n_Gamma_f)]
        ]

        # to build sum_j ((a*tau_j), (b*tau_j)) we use a trick - see Thoughts
        svfuf, svfdp, sepuf, sepdp = [
            Constant(1 / dt) * Constant(C_BJS) * (
                inner(testfunc, trialfunc)  * dxGamma
                - inner(testfunc, n_Gamma_f) * inner(trialfunc, n_Gamma_f) * dxGamma
            )
            for (testfunc, trialfunc) in [
                (Tvf, Tuf), (Tvf, Tdp), (Tep, Tuf), (Tep, Tdp)
            ]
        ]

        # a = [
        #     [adp, bpvp, 0, 0, 0, npvp],
        #     [bpvpt, -mpp, bpept, 0, 0, 0],
        #     [0, bpep, aep + sepdp, -sepuf, 0, Constant(1 / dt) * npep],
        #     [0, 0, -svfdp, af + Constant(dt) * svfuf, bf, nfvf],
        #     [0, 0, 0, bft, 0, 0],
        #     [npvpt, 0, Constant(1 / dt) * npept, nfvft, 0, 0],
        # ]

        poisson_forms = [
            (inner(testfunc, trialfunc) + inner(grad(testfunc), grad(trialfunc))) * dx
            for testfunc, trialfunc in zip(
                [vp, wp, ep, vf, wf, mu],
                [up, pp, dp, uf, pf, lbd]
            )
        ]

        a = [
            [poisson_forms[i] if j == i else 0
             for j in range(6)]
            for i in range(6)
        ]

        for i in range(6):
            for j in range(6):
                if i == j:
                    assert a[i][j] == poisson_forms[i]
                else:
                    assert a[i][j] == 0

        # quick sanity check
        N_unknowns = 6
        assert len(a) == N_unknowns
        for row in a:
            assert len(row) == N_unknowns

        # def compute_RHS(dp_prev, pp_prev, neumann_bcs, t):
        #     nf = FacetNormal(self.domain.stokes_domain)
        #     np = FacetNormal(self.domain.porous_domain)

        #     for prob_name in ["biot", "darcy", "stokes"]:  # update t in neumann bcs
        #         for expr in neumann_bcs[prob_name].keys():
        #             expr.t = t

        #     biot_neumann_terms, darcy_neumann_terms, stokes_neumann_terms = (
        #         sum(
        #             [
        #                 inner(
        #                     testfunc, neumann_bcs[prob_name][subdomain]
        #                 ) * measure(subdomain)
        #                 for subdomain in neumann_bcs[prob_name]
        #             ]
        #         ) for prob_name, testfunc, measure in zip(
        #             ["biot", "darcy", "stokes"],
        #             [ep, vp, vf],
        #             [dsDarcy, dsDarcy, dsStokes]
        #         )
        #     )

        #     s_up, s_pp, s_dp, s_uf, s_pf = self.get_source_terms()
        #     for expr in s_up, s_pp, s_dp, s_uf, s_pf:
        #         expr.t = t

        #     L_darcy = (
        #         darcy_neumann_terms
        #         + inner(vp, np) * Constant(0) * dsDarcy
        #     )
        #     L_interface = Constant(Cp) * inner(Tvp, n_Gamma_p) * dxGamma
        #     bpp = (
        #         Constant(s0 / dt) * pp_prev * wp * dxDarcy
        #         + Constant(alpha / dt) * inner(div(dp_prev), wp) * dxDarcy
        #     )
        #     L_biot = (
        #         Constant(1 / dt) * Constant(0) * inner(np, ep) * dsDarcy
        #         + Constant(Cp) * inner(Tep, n_Gamma_p) * dxGamma
        #         + biot_neumann_terms
        #     )
        #     L_stokes = Constant(0) * inner(nf, vf) * \
        #         dsStokes + stokes_neumann_terms

        #     Tdp_prev = Trace(dp_prev, self.domain.interface)
        #     L_mult = Constant(1 / dt) * inner(Tdp_prev,
        #                                       n_Gamma_p) * mu * dxGamma

        #     L = [
        #         L_darcy + L_interface + inner(s_up, vp) * dxDarcy,
        #         -bpp - inner(s_pp, wp) * dxDarcy,
        #         L_biot + Constant(1/dt) * inner(s_dp, ep) * dxDarcy,
        #         L_stokes + inner(s_uf, vf) * dxStokes,
        #         -inner(s_pf, wf) * dxStokes,
        #         L_mult

        #     ]

        #     assert len(L) == N_unknowns

        #     return L
        def compute_RHS(dp_prev, pp_prev, neumann_bcs, t):
            """Poisson"""
            nf = FacetNormal(self.domain.stokes_domain)
            np = FacetNormal(self.domain.porous_domain)

            test_funcs = vp, wp, ep, vf, wf
            normals = [np, np, np, nf, nf]
            measures = [dsDarcy] * 3 + [dsStokes]*2
            sources = self.get_source_terms()
            gradients = self.get_gradients()

            # append lambda terms
            # sources = list(sources) + [Constant(0)]
            # gradients = list(gradients) + [Constant(0)]

            
            for expr in itertools.chain(sources, gradients):
                expr.t = t
            
            L = [
                inner(v, s) * dx + inner(dot(g, n), v) * ds
                for v, n, ds, s, g in zip(test_funcs, normals, measures, sources, gradients)
            ]
            L.append(Constant(0) * mu * dxGamma)
            return L
        
        # bcs
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
            for exp in all_bc_exprs:
                exp.t = t

        bbcs = block_bc(bcs, symmetric=False)
        AA = ii_convert(AA, "")
        bbcs = bbcs.apply(
            AA
        )
        AAm = ii_convert(AA)

        w = ii_Function(self.W)

        solver = LUSolver('umfpack')  # this should be umfpack
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
        params["dt"] = 1E-4
        params["Cp"] = 1

        super(AmbartsumyanMMSProblem, self).__init__(domain, params)
        up_e, _, dp_e, uf_e, _ = self.exact_solution()

        # for prob_name, exact_sol in zip(
        #         ["darcy", "biot", "stokes"],
        #         [up_e, dp_e, uf_e]
        # ):
        #     self.add_dirichlet_bc(prob_name, 1, exact_sol)

        # for prob_name, exact_sol in zip(
        #         ["darcy", "biot", "stokes"],
        #         [up_e, dp_e, uf_e]
        # ):
        #     self.add_dirichlet_bc(prob_name, 1, exact_sol)

    def get_initial_conditions(self):
        exprs = self.exact_solution()
        for expr in exprs:
            expr.t = 0
        return exprs

    

    def exact_solution(self):
        """Return exact solution as dolfin.Expressions"""
        # up_e = Expression(
        #     (
        #         "pi * exp(t) * -cos(pi * x[0]) * cos(pi * x[1] / 2)",
        #         "pi * exp(t) * sin(pi * x[0]) * sin(pi * x[1] / 2) / 2"
        #     ), t=0, degree=5
        # )
        # pp_e = Expression(
        #     "exp(t) * sin(pi * x[0]) * cos(pi * x[1] / 2)", t=0, degree=5)
        # dp_e = Expression(
        #     (
        #         "sin(pi * t) * (-3 * x[0] + cos(x[1]))",
        #         "sin(pi * t) * (x[1] + 1)"
        #     ), t=0, degree=5
        # )

        # uf_e = Expression(
        #     (
        #         "pi * cos(pi * t) * (-3 * x[0] + cos(x[1]))",
        #         "pi * cos(pi * t) * (x[1] + 1)"
        #     ), t=0, degree=5
        # )
        # pf_e = Expression(
        #     "exp(t) * sin(pi * x[0]) * cos(pi * x[1] / 2) + 2 * pi * cos(pi * t) + Cp",
        #     t=0, degree=5, Cp=self.params["Cp"]
        # )

        up_e=Expression(
        (
          "pow(x[0], 2)*x[1]",
          "x[0]"
        ), degree=5, t=0
        )
        pp_e=Expression(
        "2*x[0] + pow(x[1], 3)", degree=5, t=0
        )
        dp_e=Expression(
        (
          "x[0] - x[1]",
          "x[0] + x[1]"
        ), degree=5, t=0
        )
        uf_e=Expression(
        (
          "2*x[1]",
          "pow(x[0], 4)"
        ), degree=5, t=0
        )
        pf_e=Expression(
        "2*x[0] + 1", degree=5, t=0
        )
        return up_e, pp_e, dp_e, uf_e, pf_e

    def get_source_terms(self):
        """ For I - Poisson"""

        # s_up = Expression( 
        #     (
        #         "-1.0/4.0*pi*(4 + 5*pow(pi, 2))*exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1])",
        #         "(1.0/8.0)*pi*(4 + 5*pow(pi, 2))*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
        #     ), degree=5, t=0
        # )
        
        # s_pp = Expression( 
        #     "exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1]) + (5.0/4.0)*pow(pi, 2)*exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])",
        #     degree=5, t=0
        # )
        
        # s_dp = Expression( 
        #     (
        #         "(-3*x[0] + 2*cos(x[1]))*sin(pi*t)",
        #         "(x[1] + 1)*sin(pi*t)"
        #     ), degree=5, t=0
        # )
        # s_uf = Expression( 
        #     (
        #         "pi*(-3*x[0] + 2*cos(x[1]))*cos(pi*t)",
        #         "pi*(x[1] + 1)*cos(pi*t)"
        #     ), degree=5, t=0
        # )
        # s_pf = Expression( 
        #     "exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1]) + (5.0/4.0)*pow(pi, 2)*exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1]) + 2*pi*cos(pi*t) + 1",
        #     degree=5, t=0
        # )
        
        s_up=Expression(
        (
          "x[1]*(pow(x[0], 2) - 2)",
          "x[0]"
        ), degree=5, t=0
        )
        s_pp=Expression(
        "2*x[0] + pow(x[1], 3) - 6*x[1]", degree=5, t=0
        )
        s_dp=Expression(
        (
          "x[0] - x[1]",
          "x[0] + x[1]"
        ), degree=5, t=0
        )
        s_uf=Expression(
        (
          "2*x[1]",
          "pow(x[0], 2)*(pow(x[0], 2) - 12)"
        ), degree=5, t=0
        )
        s_pf=Expression(
        "2*x[0] + 1", degree=5, t=0
        )

        return s_up, s_pp, s_dp, s_uf, s_pf

    def get_gradients(self):
        """ For I - Poisson"""

        g_up=Expression(
        (
          ("2*x[0]*x[1]" ,  "pow(x[0], 2)"),
          (     "1"      ,       "0"      )
        ), degree=3, t=0
        )
        g_pp=Expression(
        (
          "2",
          "3*pow(x[1], 2)"
        ), degree=3, t=0
        )
        g_dp=Expression(
        (
          ("1" ,  "-1"),
          ("1" ,  "1" )
        ), degree=3, t=0
        )
        g_uf=Expression(
        (
          (      "0"       ,        "2"       ),
          ("4*pow(x[0], 3)",        "0"       )
        ), degree=3, t=0
        )
        g_pf=Expression(
        (
          "2",
          "0"
        ), degree=3, t=0
        )

        # g_up = Expression(
        # (
        #     (      "pow(pi, 2)*exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])"      ,  "(1.0/2.0)*pow(pi, 2)*exp(t)*sin((1.0/2.0)*pi*x[1])*cos(pi*x[0])"),
        #     ("(1.0/2.0)*pow(pi, 2)*exp(t)*sin((1.0/2.0)*pi*x[1])*cos(pi*x[0])",  "(1.0/4.0)*pow(pi, 2)*exp(t)*sin(pi*x[0])*cos((1.0/2.0)*pi*x[1])")
        # ), degree=5, t=0)

        # g_pp = Expression(
        # (
        #     "pi*exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1])",
        #     "-1.0/2.0*pi*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
        # ), degree=5, t=0)

        # g_dp = Expression(
        # (
        #     (    "-3*sin(pi*t)"    ,  "-sin(x[1])*sin(pi*t)"),
        #     (          "0"           ,       "sin(pi*t)"      )
        # ), degree=5, t=0)

        # g_uf = Expression(
        # (
        #     (    "-3*pi*cos(pi*t)"    ,  "-pi*sin(x[1])*cos(pi*t)"),
        #     (             "0"             ,        "pi*cos(pi*t)"     )
        # ), degree=5, t=0)

        # g_pf = Expression(
        # (
        #     "pi*exp(t)*cos(pi*x[0])*cos((1.0/2.0)*pi*x[1])",
        #     "-1.0/2.0*pi*exp(t)*sin(pi*x[0])*sin((1.0/2.0)*pi*x[1])"
        # ), degree=5, t=0)        


        
        return g_up, g_pp, g_dp, g_uf, g_pf


    #  # original (not poisson)
    # def get_source_terms(self):
    #     ff = Expression(
    #         (
    #             "pi*(exp(t)*cos(pi*x[0])*cos(pi*x[1]/2) + cos(x[1])*cos(pi*t))",
    #             "-pi*exp(t)*sin(pi*x[0])*sin(pi*x[1]/2)/2"
    #         ), t=0, degree=5
    #     )

    #     qf = Expression("-2*pi*cos(pi*t)", t=0, degree=5)

    #     fp = Expression(
    #         (
    #             "pi*(exp(t)*cos(pi*x[0])*cos(pi*x[1]/2) + cos(x[1])*cos(pi*t))",
    #             "-pi*exp(t)*sin(pi*x[0])*sin(pi*x[1]/2)/2"
    #         ), t=0, degree=5
    #     )

    #     qp = Expression(
    #         "exp(t)*sin(pi*x[0])*cos(pi*x[1]/2) + 5*pi*pi*exp(t)*sin(pi*x[0])*cos(pi*x[1]/2)/4 - 2*pi*cos(pi*t)", degree=5, t=0
    #     )

    #     gp = Constant((0, 0))

    #     return ff, qf, fp, qp, gp
 

    def compute_errors(self, funcs, t, norm_types=None):
        """Given a list of solution values, set the time in all the exact sol
        expressions and return a list of the errornorms."""
        # TODO: this
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



def run_MMS():
    def save_to_file(things, fns=None):
        if fns is None:
            fns = ["tmp{}.pvd".format(i) for i, _ in enumerate(things)]
        for thing, fn in zip(things, fns):
            f = dolfin.File(fn)
            f << thing

    def save_errors(t, funcs, exprs, fns, norm_types):
        """For each (f, e) pair, computes the errornorm and appends it to a file."""
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
    global funcs
    result_dir = "mms_results"

    def in_dir(fn):
        return os.path.join(result_dir, fn)

    solution_fns = map(in_dir, ["up.pvd", "pp.pvd",
                                "dp.pvd", "uf.pvd", "pf.pvd", "lbd.pvd"])
    err_fns = map(in_dir, ["up_e.txt", "pp_e.txt",
                           "dp_e.txt", "uf_e.txt", "pf_e.txt"])
    # cleanup old files
    import itertools
    for fn in itertools.chain(solution_fns, err_fns):
        try:
            os.remove(fn)
        except OSError:
            pass

    Nt = 3
    for i in range(Nt + 1):
        t, funcs = solution.next()
        print "\r Done with timestep {:>3d} of {}".format(i, Nt),
        save_to_file(funcs, solution_fns)
        save_errors(t, list(funcs)[:5], problem.exact_solution(),
                    err_fns, ["l2"] * 5)

    print ""
    for func, expr, name in zip(list(funcs)[:5], problem.exact_solution(), names[:5]):
        # func2 = Function(func.function_space())
        # func2.assign(expr)
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


def make_cylinder_box_mesh(mesh_fn, h_ic, h_oc, h_b, Ri, Ro):
    import subprocess
    import os
    try:
        os.remove(mesh_fn)
    except:
        pass

    with open("cylinderbox.geo", "r") as f:
        text = "".join(f.readlines())

    text = text.replace("__H_IC__", str(h_ic), 1)
    text = text.replace("__H_OC__", str(h_oc), 1)
    text = text.replace("__H_B__", str(h_b), 1)
    text = text.replace("__Ri__", str(Ri), 1)
    text = text.replace("__Ro__", str(Ro), 1)

    tmp_geo_fn = "temp_geo.geo"
    tmp_msh_fn = "temp_geo.msh"
    with open(tmp_geo_fn, "w") as f:
        f.write(text)

    # swapangle affects the minimum allowed dihedral angle, i think
    subprocess.call(["gmsh", "-3", tmp_geo_fn, "-swapangle", "0.03"])
    subprocess.call(["dolfin-convert", tmp_msh_fn, mesh_fn])

    # os.remove(tmp_geo_fn)
    # os.remove(tmp_msh_fn)

    return Mesh(mesh_fn)


def doit():
    run_MMS()


doit()

# make_cylinder_box_mesh("mesh.xml", 0.1, 0.1, 0.1, 0.3, 0.6)
# u = Function(FunctionSpace(Mesh("mesh.xml"), "CG", 1))
# File("vizz.pvd") << u
 
