import xii
import mshr
import dolfin

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
        

class BiotStokesDomain2D(object):
    def __init__(self, N):
        EPS = 1E-3
        R = 0.25
        square_domain = mshr.Rectangle(dolfin.Point(0, 0), dolfin.Point(1, 1))
        
        stokes_subdomain = dolfin.CompiledSubDomain(
            "sqrt((x[1]-0.5) * (x[1]-0.5)) < R", R=R
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

        self._mesh = _mesh
        self._omega_p = submeshes[0]
        self._omega_f = submeshes[1]
        self._gamma = interface

        self._omega_p_markers = FacetFunction(self._omega_p, 0)
        self._omega_f_markers = FacetFunction(self._omega_f, 0)
        
    @property
    def full_domain(self):
        return self._mesh

    @property
    def stokes_domain(self):
        return self._omega_f

    @property
    def porous_domain(self):
        return self._omega_p

    @property
    def interface(self):
        return self._gamma


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
        self.set_initial_conditions()

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

    def set_initial_conditions(self):

        up = Function(self.Vp)
        pp = Function(self.Qp)
        d = Function(self.U)
        
        uf = Function(self.Vf)
        pf = Function(self.Qf)

        lbd = Function(self.X)

        self.w_prev = [up, pp, d, uf, pf, lbd]



    def get_solver(self):
        for parname in self.params.keys():
            eval("{} = self.params['{}']".format(parname, parname))
            
            
        dGamma = Measure("dx", domain=self.domain.interface)

        up, pp, d, uf, pf, lbd = map(TrialFunction, self.W)
        vp, wp, e, vf, wf, mu = map(TestFunction, self.W)

        
        Tup, Tpp, Td, Tuf, Tpf = map(lambda x: Trace(x, self.domain.interface),
                                     [up, pp, d, uf, pf, mult]
        )
        
        Tvp, Twp, Te, Tvf, Twf = map(lambda x: Trace(x, self.domain.interface),
                                     [vp, wp, e, vf, wf, multtest]
        )

        Mpp = assemble(Constant(s0/dt) * pp*wp*dx)
        
        
        
        

        # [[Ad_p, Bvp_p, 0, 0, 0, Npvpmult],
        #  [Bvp_pT, s0/dt Mpp, 

         
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



def doit2():
    n = 16
    outer_mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n, n, n)

    def domain(x):
        select = CompiledSubDomain('std::max(std::max(fabs(x[0]-x0), fabs(x[1]-x1)), fabs(x[2]-x2)) < 0.25',
                                   x0=0., x1=0., x2=0.)
        select.x0 = x[0]
        select.x1 = x[1]
        select.x2 = x[2]

        return select

    import itertools
    fs = map(domain, itertools.product(*[[-0.25, 0.25]]*3))
    
    subdomains = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim(), 0)
    # Awkward marking
    for cell in cells(outer_mesh):
        x = cell.midpoint().array()
        for tag, f in enumerate(fs, 1):
            if f.inside(x, False):
                subdomains[cell] = tag
                break

    submeshes, interface, colormap = mortar_meshes(subdomains, range(9))

#     # Subdomains
#     for cell in cells(submeshes[0]):
#         assert not any(f.inside(cell.midpoint().array(), False) for f in fs)

#     for f, mesh in zip(fs, submeshes[1:]):
#         assert all(f.inside(cell.midpoint().array(), False) for cell in cells(mesh))

#     # Interface
#     for cell in cells(interface):
#         x, y, z = cell.midpoint().array()
#         assert any((near(abs(x), 0.5) and between(y, (-0.5, 0.5)) and between(z, (-0.5, 0.5)),
#                     near(abs(y), 0.5) and between(x, (-0.5, 0.5)) and between(z, (-0.5, 0.5)),
#                     near(abs(z), 0.5) and between(x, (-0.5, 0.5)) and between(y, (-0.5, 0.5)),
#                     near(x, 0.0) and between(y, (-0.5, 0.5)) and between(z, (-0.5, 0.5)),
#                     near(y, 0.0) and between(x, (-0.5, 0.5)) and between(z, (-0.5, 0.5)),
#                     near(z, 0.0) and between(y, (-0.5, 0.5)) and between(x, (-0.5, 0.5))))
                    
#     # Map
#     tdim = interface.topology().dim()
#     for color, domains in enumerate(colormap):
#         meshes = submeshes[domains]
#         for mesh in meshes:
#             mesh.init(tdim)
#             # Each colored cell is a facet of submesh
#             for icell in SubsetIterator(interface.marking_function, color):
#                 # Index
#                 mfacet = interface.parent_entity_map[mesh.id()][tdim][icell.index()]
#                 # Actual cell
#                 mfacet = Facet(mesh, mfacet)
#                 # Now one of the facets of mcell must be icell
#                 assert near(icell.midpoint().distance(mfacet.midpoint()), 0)
doit()
