import mshr
import dolfin
from block import block_mat, block_vec, block_bc
# from petsc4py import PETSc
from dolfin import *
from xii import *
import itertools
import hsmg

class AmbartsumyanMMSDomain(object):
    """
    Stokes domain: [0, 1] x [0, 1]
    Darcy domain:  [0, 1] x [-1, 0]
    Interface:     [0, 1] x [0]

    -------
    |     |
    |  S  |
    |     |
    ------- <- I
    |     |
    |  D  |
    |     |
    -------

    """
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
        """Interface should be marked as 1. Do not set BCs there.
        Other bdy is 2"""

        stokes_markers = FacetFunction("size_t", self.stokes_domain, 0)
        porous_markers = FacetFunction("size_t", self.porous_domain, 0)

        interface_bdy = dolfin.CompiledSubDomain(
            "near(x[1], 0) && on_boundary")
        other_bdy = dolfin.CompiledSubDomain("on_boundary")

        for markers in [stokes_markers, porous_markers]:
            other_bdy.mark(markers, 2)
            interface_bdy.mark(markers, 1)

        self.stokes_bdy_markers = stokes_markers
        self.porous_bdy_markers = porous_markers

        File("stokes_bdy.pvd") << stokes_markers
        File("porous_bdy.pvd") << porous_markers
            


def function_spaces(domain):
    # biot
    Vp = FunctionSpace(domain.porous_domain, "RT", 2)
    Qp = FunctionSpace(domain.porous_domain, "DG", 1)
    U = VectorFunctionSpace(domain.porous_domain, "CG", 2)

    # stokes
    Vf = VectorFunctionSpace(domain.stokes_domain, "CG", 2)
    Qf = FunctionSpace(domain.stokes_domain, "CG", 1)

    # lagrange multiplier
    X = FunctionSpace(domain.interface, "DG", 1)

    W = [Vp, Qp, U, Vf, Qf, X]
    return W


def compute_A_P(domain):
    # names of params - all 1
    dt, alpha, alpha_BJS, s0, mu_f, mu_p, lbd_f, lbd_p, K, Cp = [float(1)]*10
    C_BJS = (mu_f * alpha_BJS) / sqrt(K)

    # measures
    dxGamma = Measure("dx", domain=domain.interface)
    dxDarcy = Measure("dx", domain=domain.porous_domain)
    dxStokes = Measure("dx", domain=domain.stokes_domain)

    # dsDarcy = Measure("ds", domain=domain.porous_domain,
    #                   subdomain_data=domain.porous_bdy_markers)
    # dsStokes = Measure("ds", domain=domain.stokes_domain,
    #                    subdomain_data=domain.stokes_bdy_markers)

    # test/trial functions
    W = function_spaces(domain)
    up, pp, dp, uf, pf, lbd = map(TrialFunction, W)
    vp, wp, ep, vf, wf, mu = map(TestFunction, W)

    # and their traces
    Tdp, Tuf = map(
        lambda x: Trace(x, domain.interface), [dp, uf]
    )
    Tep, Tvf = map(
        lambda x: Trace(x, domain.interface), [ep, vf]
    )

    # normals
    n_Gamma_f = OuterNormal(domain.interface,
                            [0.5] * domain.dimension)
    assert n_Gamma_f(Point(0.0, 0.0))[1] == -1

    n_Gamma_p = -n_Gamma_f

    Tup = Trace(up, domain.interface, restriction="-", normal=n_Gamma_f)
    Tvp = Trace(vp, domain.interface, restriction="-", normal=n_Gamma_f)


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
    bft   = - inner(div(uf), wf) * dx

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
        Constant(C_BJS) * (
            inner(testfunc, trialfunc)  * dxGamma
            - inner(testfunc, n_Gamma_f) * inner(trialfunc, n_Gamma_f) * dxGamma
        )
        for (testfunc, trialfunc) in [
            (Tvf, Tuf), (Tvf, Tdp), (Tep, Tuf), (Tep, Tdp)
        ]
    ]


    a = [
        [adp, bpvp, 0, 0, 0, npvp],
        [bpvpt, -Constant(s0/dt)*mpp, Constant(alpha/dt)*bpept, 0, 0, 0],
        [0, Constant(alpha/dt)*bpep, Constant(1/dt) * (aep + sepdp), -Constant(1/dt) * sepuf, 0, Constant(1/dt) * npep],
        [0, 0, -Constant(1/dt)*svfdp, af + svfuf, bf, nfvf],
        [0, 0, 0, bft, 0, 0],
        [npvpt, 0, Constant(1 / dt) * npept, nfvft, 0, 0],
    ]


    # quick sanity check
    N_unknowns = 6
    assert len(a) == N_unknowns
    for row in a:
        assert len(row) == N_unknowns


    ## the below cause A*P to have unbounded eigenvalues
    # homogeneous Dirichlet BCs
    up_bcs = [
        DirichletBC(
            W[0], Constant((0, 0)),
            domain.porous_bdy_markers, 2
        )
    ]

    dp_bcs = [
        DirichletBC(
            W[2], Constant((0, 0)),
            domain.porous_bdy_markers, 2
        )
    ]

    uf_bcs = [
        DirichletBC(
            W[3], Constant((0, 0)),
            domain.stokes_bdy_markers, 2
        )
    ]

    bcs = [
        up_bcs,
        [],                 # pp
        dp_bcs,
        uf_bcs,
        [],                 # pf
        []                  # lbd
    ]

    # bcs = [[] for _ in range(6)] # no bcs
    
    AA = ii_assemble(a)


    bbcs = block_bc(bcs, symmetric=True)
    AA = ii_convert(AA, "")
    AA = set_lg_map(AA)

    bbcs.apply(
        AA
    )
    AAm = ii_convert(AA)
    A = AAm.array()

    # block diagonal preconditioner
    P_up = inner(up, vp) * dxDarcy + inner(div(up), div(vp)) * dxDarcy # Hdiv
    P_pp = inner(pp, wp) * dxDarcy # L2
    P_dp = (inner(dp, ep) + inner(grad(dp), grad(ep))) * dxDarcy # H1
    P_uf = (inner(uf, vf) + inner(grad(uf), grad(vf))) * dxStokes # H1
    P_pf = (inner(pf, wf)) * dxStokes # H1
    P_up, P_pp, P_dp, P_uf, P_pf = map(
        lambda form: ii_convert(ii_assemble(form)),
        [P_up, P_pp, P_dp, P_uf, P_pf]
    )

    P_lbd = hsmg.HsNorm(W[-1], s=0.5)
    P_lbd * Function(W[-1]).vector() # enforces lazy computation

    P_lbd = P_lbd.matrix

    P = ii_convert(
        block_diag_mat([P_up, P_pp, P_dp, P_uf, P_pf, P_lbd]), ""
    )

    # TODO: should I use the bbcs object returned from the call to bbcs.apply() above here?
    # or would that only be for if i wanted to apply to vectors?
    bbcs.apply(
        P
    )
    P = ii_convert(P).array()
    return A, P



# eigenvalue experiment
Ps = {}
As = {}
eigses = {}
Ns = [4, 8, 16]
for N in Ns:
    domain = AmbartsumyanMMSDomain(N)
    
    A, P = compute_A_P(domain)

    Ps[N] = P
    As[N] = A
    import scipy

    import numpy as np
    # print np.max(A), np.min(A)
    # print np.max(P), np.min(P)
    
    
    eigs = scipy.linalg.eigh(A, P, eigvals_only=True)

    eigs = sorted(map(abs, eigs))

    eigses[N] = eigs
    
    print "N={}: min = {:.6f}, max = {:.6f}".format(N, min(eigs), max(eigs))

print "\nResults:"
for N in Ns:
    eigs = eigses[N]
    print "N={:>3d}: min = {:.6f}, max = {:.6f}".format(N, min(eigs), max(eigs))
