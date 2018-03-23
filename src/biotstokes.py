import xii
import mshr
import dolfin
from block import block_mat, block_vec, block_bc
from petsc4py import PETSc
from dolfin import *
from xii import *

class BiotStokesDomain3D(object):
    def __init__(self, N):
        EPS = 1E-3
        R = 0.25
        box_domain = mshr.Box(dolfin.Point(0, 0, 0), dolfin.Point(1, 1, 1))
        _mesh = mshr.generate_mesh(box_domain, N)

        stokes_subdomain = dolfin.CompiledSubDomain(
            "sqrt((x[0]-0.5) * (x[0]-0.5) + (x[1]-0.5) * (x[1]-0.5)) < R", R=R
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
        self.stokes_domain = submeshes[0]
        self.porous_domain = submeshes[1]
        self.interface = interface

        self.mark_boundary()
        

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

        self.add_neumann_bc("stokes", 1, 5)
        self.add_neumann_bc("stokes", 2, 10)
        
    def add_dirichlet_bc(self, problem_name, subdomain_id, value):
        bc_dict = self._dirichlet_bcs[problem_name]
        if subdomain_id in bc_dict:
            del bc_dict[subdomain_id]
            
        bc_dict[subdomain_id] = value

    def add_neumann_bc(self, problem_name, subdomain_id, value):
        bc_dict = self._neumann_bcs[problem_name]
        if subdomain_id in bc_dict:
            del bc_dict[subdomain_id]
            
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
        
        C_BJS = (mu_f * alpha_BJS) / sqrt(K)    

        # names of things needed to build matrices
        dxGamma = Measure("dx", domain=self.domain.interface)
        
        up, pp, dp, uf, pf, lbd = map(TrialFunction, self.W)
        vp, wp, ep, vf, wf, mu = map(TestFunction, self.W)
        
        # thank you, Miro!
        Tup, Tdp, Tuf = map(lambda x: Trace(x, self.domain.interface),
                            [up, dp, uf]
        )
        
        Tvp, Tep, Tvf = map(lambda x: Trace(x, self.domain.interface),
                            [vp, ep, vf]
        )

        nf = OuterNormal(self.domain.interface, [0.5, 0.5, 0.5]) # todo: verify that this is actually the right normal
        np = -nf
        
        # build a bunch of matrices
        
        Mpp = assemble(Constant(s0/dt) * pp * wp*dx)
        Adp = assemble(Constant(mu_p/K) * inner(up, vp)*dx)
        Aep = assemble(
            Constant(mu_p/dt) * inner(sym(grad(dp)), sym(grad(ep)))*dx
            + Constant(lbd_p) * inner(div(dp), div(ep))  * dx
        )
        Af = assemble(Constant(2*mu_f) * inner(sym(grad(uf)), sym(grad(vf))) * dx)

        Bpvp = assemble(- inner(div(vp), pp)*dx)
        Bpep = assemble(- Constant(alpha/dt) * inner(div(ep), pp)*dx)
        Bf = assemble(- inner(div(vf), pf) * dx)

        # matrices living on the interface
        Npvp, Npep, Nfvf = [ii_convert(ii_assemble(lbd * dot(testfunc, n) * dxGamma))
                            for (testfunc, n) in [(Tvp, np), (Tep, np), (Tvf, nf)]]

        # to build sum_j ((a*tau_j), (b*tau_j)) we use a trick - see Thoughts
        Svfuf, Svfdp, Sepuf, Sepdp = [
            ii_convert(ii_assemble(
                inner(testfunc, trialfunc) * Constant(1/dt) * dxGamma
                - inner(testfunc, nf) * inner(trialfunc, nf) * Constant(1/dt) * dxGamma
            ))
            for (testfunc, trialfunc) in [
                    (Tvf, Tuf), (Tvf, Tdp), (Tep, Tuf), (Tep, Tdp)
            ]
        ]
        
        
        ## this matrix assembly is kind of memory-wasteful, so rewrite for bigger problem
        def _scale(matrix, c):
            """matrix*c. More specifically, takes a PETScMatrix (as returned from assemble()) 
            and returns the matrix scaled by a factor c (does NOT modify original)"""
            
            new_matrix = as_backend_type(matrix).mat().copy()
            new_matrix.scale(c)
            return PETScMatrix(new_matrix)
        
        def _sum(matrix1, matrix2):
            """Same as the above, but matrix1+matrix2"""
            matrix1, matrix2 = [as_backend_type(m).mat()
                                for m in (matrix1, matrix2)]
            return PETScMatrix(matrix1 + matrix2)

        def _t(matrix):
            """Same as the above, but transpose"""
            new_matrix = PETSc.Mat()
            as_backend_type(matrix).mat().transpose(new_matrix)
            return PETScMatrix(new_matrix)

        AA = [
            [Adp, Bpvp, 0, 0, 0, Npvp],
            [_t(Bpvp), Mpp, _t(Bpep), 0, 0, 0],
            [0, _t(Bpep), _sum(Aep, Sepdp), _scale(Sepuf, -1), 0, _scale(Npep, 1/dt)],
            [0, 0, _scale(Svfdp, -1), _sum(Af, _scale(Svfuf, dt)), Bf, Nfvf],
            [0, 0, 0, _t(Bf), 0, 0],
            [_t(Npvp), 0, _t(_scale(Npep, 1/dt)), _t(Nfvf), 0, 0],
        ]

        # block_matrix probably handles this
        assert len(AA) == 6
        for row in AA:
            assert len(row) == 6
            
        AA = block_mat(AA)
        # AA = set_lg_rc_map(AA, self.W)

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
        bbcs = block_bc(bcs, symmetric=False).apply(AA) # todo: can i put symmetric=True here?

        # can make the solver now
        A = ii_convert(AA)
        solver = LUSolver('mumps') # this should be umfpack
        solver.set_operator(A)
        solver.parameters['reuse_factorization'] = True

        # need indices to put data from function back into rhs
        dims = [space.dim() for space in W]
        indices = [
            PETSc.IS().createGeneral(
                range(sum(dims[:n]), sum(dims[:n]) + dims[n])
            )
            for n in range(len(dims))
        ]
        
        # set up initial conditions
        funcs = map(Function, self.W)
        vecs = [f.vec() for f in funcs]

        block_vec = block_vec(vecs)
        bbcs.apply(block_vec)
        
        x = block_to_dolfin(block_vec)
        x_vec = as_backend_type(x).vec()

        # alright, we're all set
        t = 0
        yield [up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev]
        
        while True:
            # solve for next time step
            t += dt
            # todo: set t for BCs if time-dependent and if so, reapply to A

            # todo: get new RHS
            bb = None
            bbcs.apply(bb)
            solver.solve(x, block_to_dolfin(bb))

            # now put the x vector back into the dolfin.Functions
            for idx, vector, function in indices, vecs, funcs:
                x_vec.getSubVector(idx, vector) # extract from x_vec and store in vector
                # the below shouldn't be necessary, but somehow it is
                function.vector().zero()
                # the below isn't just 'vector', for some reason?
                function_vec = as_backend_type(function.vector()).vec()
                vector.copy(function_vec)

            yield funcs
        
        
    


def doit():
    N = 10
    # domain = BiotStokesDomain2D(N)
    domain = BiotStokesDomain3D(N)
    save_to_file(
        [domain.stokes_domain, domain.porous_domain,
         domain.interface, domain.full_domain],
        ["stokes.pvd", "biot.pvd", "interface.pvd", "full.pvd"],
    )
    
    problem = BiotStokesProblem(domain, {})
    solution = problem.get_solver()

    print("First call to nesxt()")
    solution.next()
    print("Done with first call to nesxt() !!")
    
def save_to_file(things, fns=None):
    if fns is None:
        fns = ["tmp{}.pvd".format(i) for i, _ in enumerate(things)]
    for thing, fn in zip(things, fns):
        f = dolfin.File(fn)
        f << thing


doit()
