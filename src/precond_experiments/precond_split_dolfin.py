import pandas as pd
import pickle
import scipy
import numpy as np
import mshr
import dolfin
from block import block_mat, block_vec, block_bc
# from petsc4py import PETSc
from dolfin import *
from xii import *
import itertools
import hsmg
import matplotlib.pyplot as plt

dolfin.set_log_level(dolfin.ERROR)

brain_params = {
    "alpha": 0.9995,
    "mu_f": 0.8E-3,
    "lbd_p": 7142.,        # gets bad if bumped up an order of magnitude
    "mu_p": 1786,
    "K": 1E-5,                 # gets bad at 1E-1, needs to be 1E-5
    "alpha_BJS": 1E-2,      # gets bad at 1E-3
    "dt": 1E-3,
    "s0": 1/7142.,
    "Cp": 0
}

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

    @property                   # interface normal pointing outwards from fluid domain
    def interface_normal_f(self):
        return OuterNormal(self.interface, [0.5] * self.dimension)
    @property
    def interface_normal_p(self):
        return OuterNormal(self.interface, [-0.5] * self.dimension)


    def mark_boundary(self):
        """Interface should be marked as 1. Do not set BCs there.
        Other bdy is 2"""

        stokes_markers = MeshFunction("size_t", self.stokes_domain, 1, 0)
        porous_markers = MeshFunction("size_t", self.porous_domain, 1, 0)

        interface_bdy = dolfin.CompiledSubDomain(
            "near(x[1], 0) && on_boundary")
        other_bdy = dolfin.CompiledSubDomain("on_boundary")

        for markers in [stokes_markers, porous_markers]:
            other_bdy.mark(markers, 2)
            interface_bdy.mark(markers, 1)

        self.stokes_bdy_markers = stokes_markers
        self.porous_bdy_markers = porous_markers



def function_spaces(domain):
    # biot
    Vp = FunctionSpace(domain.porous_domain, "RT", 2)
    Qp = FunctionSpace(domain.porous_domain, "DG", 1)
    U = VectorFunctionSpace(domain.porous_domain, "CG", 2)

    # stokes
    Vf = VectorFunctionSpace(domain.stokes_domain, "CG", 2)
    Qf = FunctionSpace(domain.stokes_domain, "CG", 1)

    # lagrange multiplier
    X = FunctionSpace(domain.interface, "CG", 1)

    W = [Vp, Qp, U, Vf, Qf, X]
    # W = [Vp, Qp, U, Vf, Qf]

    return W


def compute_A_P_BDS(domain, params):
    """Computes A, P for Biot/Darcy/Stokes problem."""
    
    # # names of params - all 1
    # dt, alpha, alpha_BJS, s0, mu_f, mu_p, lbd_f, lbd_p, K, Cp = [float(1)]*10
    dt = params["dt"]
    alpha = params["alpha"]
    alpha_BJS = params["alpha_BJS"]
    s0 = params["s0"]
    mu_f = params["mu_f"]
    mu_p = params["mu_p"]
    lbd_p = params["lbd_p"]
    K = params["K"]
    Cp = params["Cp"]

    

    
    C_BJS = (mu_f * alpha_BJS) / sqrt(K)

    # measures
    dxGamma = Measure("dx", domain=domain.interface)
    dxDarcy = Measure("dx", domain=domain.porous_domain)
    dxStokes = Measure("dx", domain=domain.stokes_domain)

    dsDarcy = Measure("ds", domain=domain.porous_domain,
                      subdomain_data=domain.porous_bdy_markers)
    dsStokes = Measure("ds", domain=domain.stokes_domain,
                       subdomain_data=domain.stokes_bdy_markers)

    # test/trial functions
    W = function_spaces(domain)
    up, pp, dp, uf, pf, lbd = map(TrialFunction, W)
    vp, wp, ep, vf, wf, mu = map(TestFunction, W)

    # and their traces
    # Tdp, Tuf = map(
    #     lambda x: Trace(x, domain.interface), [dp, uf]
    # )
    # Tep, Tvf = map(
    #     lambda x: Trace(x, domain.interface), [ep, vf]
    # )

    # # normals
    # n_Gamma_f = OuterNormal(domain.interface,
    #                         [0.5] * domain.dimension)
    # assert n_Gamma_f(Point(0.0, 0.0))[1] == -1

    # n_Gamma_p = -n_Gamma_f

    # Tup = Trace(up, domain.interface, restriction="-", normal=n_Gamma_f)
    # Tvp = Trace(vp, domain.interface, restriction="-", normal=n_Gamma_f)


    # a bunch of forms
    # stokes 
    af =  Constant(2 * mu_f) * inner(sym(grad(uf)), sym(grad(vf))) * dxStokes
    bf    = - inner(div(vf), pf) * dxStokes
    bft   = - inner(div(uf), wf) * dxStokes

    # biot
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

    Co = Constant

    # biot
    # a = [
    #         [adp, bpvp, 0],
    #         [bpvpt, -Co(s0 / dt) * mpp, Co(alpha / dt) * bpept],
    #         [0, Constant(alpha / dt) * bpep, Co(1 / dt) * aep],
    # ]
    
    # biot
    # a = [
    #     [af, bf],
    #     [bft, 0],
    # ]

    # both w/o multiplier
    a = [
            [adp, bpvp, 0, 0, 0],
            [bpvpt, -Co(s0 / dt) * mpp, Co(alpha / dt) * bpept, 0, 0],
            [0, Constant(alpha / dt) * bpep, Co(1 / dt) * aep, 0, 0],
            [0, 0, 0, af, bf],
            [0, 0, 0, bft, 0],
    ]

    # # with multiplier
    # a = [
    #         [adp, bpvp, 0, 0, 0, 0],
    #         [bpvpt, -Co(s0 / dt) * mpp, Co(alpha / dt) * bpept, 0, 0, 0],
    #         [0, Constant(alpha / dt) * bpep, Co(1 / dt) * aep, 0, 0, 0],
    #         [0, 0, 0, af, bf, 0],
    #         [0, 0, 0, bft, 0, 0],
    #         [0, 0, 0, 0, 0, inner(mu, lbd) * dxGamma]
    # ]

    
    # homogeneous Dirichlet BCs
    up_bcs = [
        DirichletBC(
            W[0], Constant((0, 0)),
            domain.porous_bdy_markers, i
        ) for i in [1, 2]
    ]

    dp_bcs = [
        DirichletBC(
            W[2], Constant((0, 0)),
            domain.porous_bdy_markers, i
        ) for i in [1, 2]
    ]

    uf_bcs = [
        DirichletBC(
            W[3], Constant((0, 0)),
            domain.stokes_bdy_markers, i
        ) for i in [1]          # if there are dirichlet BCs all over, pressure gets underdetermined
    ]

    bcs = [
        up_bcs,
        [],                 # pp
        dp_bcs,
        uf_bcs,
        [],                 # pf
    ]

    AA = ii_assemble(a)

    bbcs = block_bc(bcs, symmetric=True)
    AA = ii_convert(AA, "")
    # AA = set_lg_map(AA)

    bbcs.apply(
        AA
    )
    
    A = ii_convert(AA).array()
    # A = AAm.array()

    # weighted block diagonal preconditioner
    P_up = Constant(mu_p/K) * inner(up, vp) * dxDarcy
    P_pp = Constant(s0/dt + 1/(mu_p+lbd_p)) * inner(pp, wp) * dxDarcy + Constant(K/mu_p) * inner(grad(pp), grad(wp)) * dxDarcy
    P_dp = (Constant(mu_p/dt) * inner(grad(dp), grad(ep))) * dxDarcy
    
    P_uf = (Constant(mu_f) * inner(grad(uf), grad(vf))) * dxStokes
    P_pf = (Constant(1/mu_f) * inner(pf, wf)) * dxStokes
    
    P_up, P_pp, P_dp, P_uf, P_pf = map(
    # P_up, P_pp, P_dp = map(
    # P_uf, P_pf = map(
        lambda form: ii_convert(ii_assemble(form)),
        [
            P_up, P_pp, P_dp,
            P_uf, P_pf
        ]
    )

    P_lbd = hsmg.HsNorm(W[-1], s=0.5)
    P_lbd * Function(W[-1]).vector() # enforces lazy computation

    P_lbd = P_lbd.matrix
    
    P = ii_convert(
        block_diag_mat([
            P_up, P_pp, P_dp,
            P_uf, P_pf,
            # P_lbd
        ]), ""
    )

    # TODO: should I use the bbcs object returned from the call to bbcs.apply() above here?
    # or would that only be for if i wanted to apply to vectors?
    bbcs.apply(
        P
    )
    P = ii_convert(P).array()
    return A, P

def condition_number(A, P):
    eigs = scipy.linalg.eigh(A, P, eigvals_only=True)
    eigs = sorted(map(abs, eigs))
    return max(eigs)/min(eigs)

N=4
domain = AmbartsumyanMMSDomain(N)
A, P = compute_A_P_BDS(domain, brain_params)
eigs = scipy.linalg.eigh(A, P, eigvals_only=True)
eigs = sorted(map(abs, eigs))

condition_number = max(eigs)/min(eigs)

print "K: {:.3e}".format(condition_number)
