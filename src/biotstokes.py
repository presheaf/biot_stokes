# import xii
import mshr
import dolfin
from block import block_mat, block_vec, block_bc
# from petsc4py import PETSc
from dolfin import *
from xii import *


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

        submeshes, interface, _ = mortar_meshes(subdomains, range(2), strict=True, tol=EPS)

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
        EPS = 1E-3
        R = 0.3
        # rectangle_domain = mshr.Rectangle(dolfin.Point(0, -1), dolfin.Point(1, 1))
        top_half = mshr.Rectangle(dolfin.Point(0, 0), dolfin.Point(1, 1))
        bottom_half = mshr.Rectangle(dolfin.Point(0, -1), dolfin.Point(1, 0))
        _topmesh = mshr.generate_mesh(top_half, N)
        _botmesh = mshr.generate_mesh(bottom_half, N)

        _mesh = mshr.DolfinMeshUtils.merge_meshes(_topmesh, _botmesh)
        # _mesh = mshr.generate_mesh(rectangle_domain, N)
        # _mesh = UnitSquareMesh(N, N)

        stokes_subdomain = dolfin.CompiledSubDomain(
            "x[1] > 0", R=R, eps=EPS
        )

        subdomains = MeshFunction('size_t', _mesh, _mesh.topology().dim(), 0)

        # Awkward marking
        for cell in cells(_mesh):
            x = cell.midpoint().array()
            if stokes_subdomain.inside(x, False):
                subdomains[cell] = 1
            else:
                subdomains[cell] = 0

        # strict = False here because in 2D, the interface is actually disconnected
        submeshes, interface, _ = mortar_meshes(subdomains, range(2), strict=True, tol=EPS)

        self.full_domain = _mesh
        self.stokes_domain = submeshes[1]
        self.porous_domain = submeshes[0]
        self.interface = interface

        self.mark_boundary()

    @property
    def dimension(self):
        return 2

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
            "stokes": {0: 0},
            "biot": {0: 0},
            "darcy": {0: 0}
        }

        self.make_function_spaces()


        
    def add_dirichlet_bc(self, problem_name, subdomain_id, value):
        bc_dict = self._dirichlet_bcs[problem_name]
        # if subdomain_id in bc_dict:
        #     del bc_dict[subdomain_id]
            
        bc_dict[subdomain_id] = value

    def add_neumann_bc(self, problem_name, subdomain_id, value):
        bc_dict = self._neumann_bcs[problem_name]
        # if subdomain_id in bc_dict:
        #     del bc_dict[subdomain_id]
            
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
        up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev = map(Function, self.W)
        
        # thank you, Miro!
        Tup, Tdp, Tuf = map(lambda x: Trace(x, self.domain.interface),
                            [up, dp, uf]
        )
        
        Tvp, Tep, Tvf = map(lambda x: Trace(x, self.domain.interface),
                            [vp, ep, vf]
        )

        # todo: verify that this is actually the right normal
        n_Gamma_f = OuterNormal(self.domain.interface, [0.5] * self.domain.dimension)
        n_Gamma_p = -n_Gamma_f
        
        # build a bunch of matrices
        
        mpp = Constant(s0 / dt) * pp * wp*dx
        adp = Constant(mu_p / K) * inner(up, vp)*dx
        aep = (
            Constant(mu_p / dt) * inner(sym(grad(dp)), sym(grad(ep))) * dx
            + Constant(lbd_p) * inner(div(dp), div(ep))  * dx
        )
        af = Constant(2*mu_f) * inner(sym(grad(uf)), sym(grad(vf))) * dx

        bpvp = - inner(div(vp), pp) * dx
        bpvpt = - inner(div(up), wp) * dx
        bpep = - Constant(alpha/dt) * inner(div(ep), pp) * dx
        bpept = - Constant(alpha/dt) * inner(div(dp), wp) * dx
        bf = - inner(div(vf), pf) * dx
        bft = - inner(div(uf), wf) * dx

        # matrices living on the interface
        npvp, npep, nfvf = [lbd * dot(testfunc, n) * dxGamma
                            for (testfunc, n) in [(Tvp, n_Gamma_p), (Tep, n_Gamma_p), (Tvf, n_Gamma_f)]]
        npvpt, npept, nfvft = [mu * dot(trialfunc, n) * dxGamma
                            for (trialfunc, n) in [(Tup, n_Gamma_p), (Tdp, n_Gamma_p), (Tuf, n_Gamma_f)]]



        # to build sum_j ((a*tau_j), (b*tau_j)) we use a trick - see Thoughts
        svfuf, svfdp, sepuf, sepdp = [
            
            inner(testfunc, trialfunc) * Constant(1 / dt) * dxGamma
            - inner(testfunc, n_Gamma_f) * inner(trialfunc, n_Gamma_f) * Constant(1 / dt) * dxGamma
            for (testfunc, trialfunc) in [
                    (Tvf, Tuf), (Tvf, Tdp), (Tep, Tuf), (Tep, Tdp)
            ]
        ]
        
        a = [
            [adp, bpvp, 0, 0, 0, npvp],
            [bpvpt, mpp, bpept, 0, 0, 0],
            [0, bpep, aep+ sepdp, -sepuf, 0, Constant(1/dt) * npep],
            [0, 0, -svfdp, af+ Constant(dt)*svfuf, bf, nfvf],
            [0, 0, 0, bft, 0, 0],
            [npvpt, 0, Constant(1/dt) * npept , nfvft, 0, 0],
        ]



        # quick sanity check
        N_unknowns = 6
        assert len(a) == N_unknowns
        for row in a:
            assert len(row) == N_unknowns

        def compute_RHS(dp_prev, neumann_bcs):
            # TODO: put Neumann BCs here

            nf = FacetNormal(self.domain.stokes_domain)
            np = FacetNormal(self.domain.porous_domain)

            L_pp = inner(vp, np) * Constant(0) * dsDarcy 
            L_interface = Constant(Cp) * inner(Tvp, n_Gamma_p) * dxGamma
            bpp = (
                Constant(s0 / dt) * pp_prev * wp * dxDarcy
                - Constant(alpha / dt) * inner(div(dp_prev), wp) * dxDarcy
            )
            L_darcy = Constant(1 / dt) * Constant(0) * inner(np, ep) * dsDarcy
            L_stokes = Constant(0) * inner(nf , vf) * dsStokes

            Tdp_prev = Trace(dp_prev, self.domain.interface)
            L_mult = Constant(1 / dt) * inner(Tdp_prev, n_Gamma_p) * mu * dxGamma   

            L = [
                L_pp + L_interface,
                -bpp,
                L_darcy,
                L_stokes,
                Constant(0) * wf * dsStokes,
                L_mult

            ]

            assert len(L) == N_unknowns
            
            return L

            
        # bcs 
        up_bcs = [
            DirichletBC(
                self.W[0], self._dirichlet_bcs["darcy"][subdomain_num],
                self.porous_bdy_markers, subdomain_num
            )
            for subdomain_num in self._dirichlet_bcs["darcy"]
        ]

        dp_bcs = [
            DirichletBC(
                self.W[2], self._dirichlet_bcs["biot"][subdomain_num],
                self.porous_bdy_markers, subdomain_num
            )
            for subdomain_num in self._dirichlet_bcs["biot"]
        ]

        uf_bcs = [
            DirichletBC(
                self.W[3], self._dirichlet_bcs["stokes"][subdomain_num],
                self.stokes_bdy_markers, subdomain_num
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
        bbcs = block_bc(bcs, symmetric=False).apply(AA) # todo: can i put symmetric=True here?
        AAm = ii_convert(AA)

        w = ii_Function(self.W)
        
        solver = LUSolver('umfpack') # this should be umfpack
        solver.set_operator(AAm)
        solver.parameters['reuse_factorization'] = True

        # alright, we're all set
        t = 0
        yield [up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev]
        
        while True:
            # solve for next time step
            t += dt
            # todo: set t for BCs if time-dependent and if so, reapply to A

            L = compute_RHS(dp_prev, self._neumann_bcs)
            bb = ii_assemble(L)
            bbcs.apply(bb)
            bbm = ii_convert(bb)            
            
            solver.solve(w.vector(), bbm)

            for i, func in enumerate([up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev]):
                func.assign(w[i])

            yield w


def doit():
    # N = 10
    hi, ho = 0.1, 0.1
    # domain = AmbartsumyanMMSDomain(N)
    domain = CylinderBoxDomain3D(hi, ho)

    problem = BiotStokesProblem(domain, {})
    solution = problem.get_solver()
    problem.add_neumann_bc("stokes", 1, 5)
    problem.add_neumann_bc("stokes", 2, 10)

    
    FunctionSpace
    print("First call to nesxt()")
    funcs = solution.next()
    print("Done with first call to next() !!")
    # solution.next()
    # funcs = solution.next()
    # print("2 and 3 done!!")
    
    def save_to_file(things, fns=None):
        if fns is None:
            fns = ["tmp{}.pvd".format(i) for i, _ in enumerate(things)]
        for thing, fn in zip(things, fns):
            f = dolfin.File(fn)
            f << thing

    save_to_file(
        funcs,
        ["up.pvd", "pp.pvd", "dp.pvd", "uf.pvd", "pf.pvd", "lbd.pvd"],
    )
    save_to_file(
        [domain.stokes_domain, domain.porous_domain,
         domain.interface, domain.full_domain],
        ["stokes.pvd", "biot.pvd", "interface.pvd", "full.pvd"],
    )




def make_cylinder_box_mesh(mesh_fn, hi, ho, R):
    import subprocess, os
    try:
        os.remove(mesh_fn)
    except:
        pass
    
    with open("cylinderboxtemplate.geo", "r") as f:
        text = "".join(f.readlines())

    
    text = text.replace("__HI__", str(hi), 1)
    text = text.replace("__HO__", str(ho), 1)
    text = text.replace("__R__", str(R), 1)
    
    tmp_geo_fn = "temp_geo.geo"
    tmp_msh_fn = "temp_geo.msh"
    with open(tmp_geo_fn, "w") as f:
        f.write(text)

    subprocess.call(["gmsh", "-3", tmp_geo_fn])
    subprocess.call(["dolfin-convert", tmp_msh_fn, mesh_fn])

    os.remove(tmp_geo_fn)
    os.remove(tmp_msh_fn)

    return Mesh(mesh_fn)


    
    
    

doit()
