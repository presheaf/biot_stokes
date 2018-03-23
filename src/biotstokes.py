import xii
import mshr
import dolfin
from block import block_mat, block_vec, block_bc

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

        self.mark_boundary
        

    def mark_boundary(self):
        stokes_markers = FacetFunction("size_t", self.stokes_domain, 0)
        porous_markers = FacetFunction("size_t", self.porous_domain, 0)

        full_bdy = dolfin.CompiledSubDomain("on_boundary")
        left_bdy = dolfin.CompiledSubDomain("near(x[0], 0) && on_boundary")
        right_bdy = dolfin.CompiledSubDomain("near(x[0], 1) && on_boundary")

        for markers in [stokes_markers, porous_markers]:
            full_bdy.mark(markers, 1)
            left_bdy.mark(markers, 2)
            right_bdy.mark(markers, 3)

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

        self.add_neumann_bc("stokes", 2, 10)
        self.add_neumann_bc("stokes", 1, 5)
        
    def add_dirichlet_bc(self, problem_name, subdomain_id, value):
        if subdomain_id in self._neumann_bc[problem_name]:
            del self._neumann_bc[problem_name][subdomain_id]
        self._dirichlet_bcs[problem_name][subdomain_id] = value

    def add_neumann_bc(self, problem_name, subdomain_id, value):
        if subdomain_id in self._neumann_bc[problem_name]:
            del self._neumann_bc[problem_name][subdomain_id]
        self._dirichlet_bcs[problem_name][subdomain_id] = value

    def make_function_spaces(self):
        # biot
        Vp = VectorFunctionSpace(self.domain.porous_domain, "RT", 1)
        Qp = FunctionSpace(self.domain.porous_domain, "DG", 1)
        U = VectorFunctionSpace(self.domain.porous_domain, "CG", 2)

        # stokes
        Vf = FunctionSpace(self.domain.stokes_domain, "CG", 2)
        Qf = FunctionSpace(self.domain.stokes_domain, "CG", 1)
        
        # lagrange multiplier
        X = FunctionSpace(self.domain.interface, "DG", 1)

        self.W = [Vp, Qp, U, Vf, Qf, X]


    def get_solver(self):
        """Returns an iterator over solution values. Values are returned as a 
        list of Functions, with the ordering being [up, pp, dp, uf, pf, lbd]. 
        First returned value is initial conditions (zero)."""
        
        # names of params
        for parname in self.params.keys():
            eval("{} = self.params['{}']".format(parname, parname))
        C_BJS = (mu_f * alpha_BJS) / sqrt(K)    

        # names of things needed to build matrices
        dGamma = Measure("dx", domain=self.domain.interface)
        up, pp, dp, uf, pf, lbd = map(TrialFunction, self.W)
        vp, wp, ep, vf, wf, mu = map(TestFunction, self.W)
        
        # thank you, Miro!
        Tup, Tdp, Tuf = map(lambda x: Trace(x, self.domain.interface),
                            [up, dp, uf]
        )
        
        Tvp, Tep, Tvf = map(lambda x: Trace(x, self.domain.interface),
                            [vp, ep, vf]
        )
        nf = OuterNormal(domain.interface, [0.5, 0.5, 0.5]) # todo: verify that this is actually the right normal
        np = -nf

        
        # build a bunch of matrices
        Mpp = assemble(Constant(s0/dt) * pp * wp*dx)
        Adp = assemble(Constant(mu_p/K) * inner(up, vp)*dx)
        Aep = assemble(
            Constant(mu_p/dt) * inner(sym(dp), sym(ep))*dx
            + Constant(lbd_p) * inner(div(dp), div(ep))
        )
        Bpvp = assemble(- inner(div(vp), pp)*dx)
        Bpep = assemble(- Constant(alpha/dt) * inner(div(ep), pp)*dx)
        Bf = assemble(- inner(div(vf), pf) * dx)
        

        Npvp, Npep, Nfvf = [assemble(lbd * inner(testfunc, n) * dGamma)
                            for (testfunc, n) in [(Tvp, np), (Tep, np), (Tvf, nf)]]

        # to build sum_j ((a*tau_j), (b*tau_j)) we use a trick - see Thoughts
        Svfuf, Svfdp, Sepuf, Sepdp = [
            assemble(
                (inner(testfunc, trialfunc)
                 - inner(testfunc, nf) * inner(trialfunc, nf)
                ) * Constant(1/dt) * dGamma
            )
            for (testfunc, trialfunc) in [(Tvf, Tuf), (Tvf, Tdp), (Tep, Tuf), (Tep, Tdp)]
        ]
        


        # this matrix assembly is kind of memory-wasteful, so rewrite for bigger problem
        def _scale(matrix, c):
            """matrix*c. More specifically, takes a PETScMatrix (as returned from assemble()) 
            and returns the matrix scaled by a factor c (does NOT modify original)"""
            
            new_matrix = as_backend_type(matrix).mat().copy()
            new_matrix.scale(c)
            return PETScMatrix(new_matrix)
        
        def _sum(matrix1, matrix2):
            """Same as the above, but matrix1+matrix2"""
            new_matrix = as_backend_type(matrix1).mat().copy()
            new_matrix.axpy(1., matrix2)
            return PETScMatrix(new_matrix)

        def _t(matrix):
            """Same as the above, but transpose"""
            new_matrix = as_backend_type(matrix1).mat().copy()
            new_matrix.transpose()
            return PETScMatrix(new_matrix)

        
        AA = block_mat(
            [
                [Adp, Bpvp, 0, 0, 0, Npvp],
                [_t(Bpvp), Mpp, _t(Bpep), 0, 0, 0],
                [0, _t(Bpep), _sum(Aep, Sepdp), _scale(Sepuf, -1), 0, _scale(Npep, 1/dt)],
                [0, 0, _scale(Svfdp, -1), _sum(Af, _scale(Svfuf, dt)), Bf, Nfvf]
                [0, 0, 0, _t(Bf), 0, 0],
                [_t(Npvp), 0, _t(_scale(Npep, 1/dt)), _t(Nfvf), 0, 0],

            ]
        )
        AA = set_lg_rc_map(AA, self.W)

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
        bbcs = block_bcs(bcs, symmetric=False).apply(AA) # todo: can i put symmetric=True here?

        
        dims = [space.dim() for space in W]
        indices = [
            PETSc.IS().createGeneral(
                range(sum(dims[:n]), sum(dims[:n]) + dims[n])
            )
            for n in range(len(dims))
        ]
        
        # set up initial conditions
        up_prev, pp_prev, dp_prev, uf_prev, pf_prev, lbd_prev = map(Function, self.W)
        
        # define RHS
        def rhs(w_prev):
            # todo: make a func which returns a vector bb
            pass


        A = block_to_dolfin(AA)
        solver = LUSolver('mumps') # this should be umfpack
        solver.set_operator(A)
        solver.parameters['reuse_factorization'] = True

        yield self.w
        
        while True:
            # solve for next time step
            print "solved"
            yield
        
        
    


def doit():
    N = 20
    # domain = BiotStokesDomain2D(N)
    domain = BiotStokesDomain3D(N)
    save_to_file(
        [domain.stokes_domain, domain.porous_domain,
         domain.interface, domain.full_domain],
        ["stokes.pvd", "biot.pvd", "interface.pvd", "full.pvd"],
    )
    problem = BiotStokesProblem(domain, {})
    
def save_to_file(things, fns=None):
    if fns is None:
        fns = ["tmp{}.pvd".format(i) for i, _ in enumerate(things)]
    for thing, fn in zip(things, fns):
        f = dolfin.File(fn)
        f << thing


doit()
