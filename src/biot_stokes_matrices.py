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
    # a = [
    #     [adp, bpvp, 0, 0, 0, npvp],
    #     [bpvpt, -Constant(s0/dt)*mpp, Constant(alpha/dt)*bpept, 0, 0, 0],
    #     [0, Constant(alpha/dt)*bpep, Constant(1/dt) * (aep + sepdp), -Constant(1/dt) * sepuf, 0, Constant(1/dt) * npep],
    #     [0, 0, -Constant(1/dt)*svfdp, af + svfuf, bf, nfvf],
    #     [0, 0, 0, bft, 0, 0],
    #     [npvpt, 0, Constant(1 / dt) * npept, nfvft, 0, 0],
    # ]


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

    
    
    # AAm = ii_convert(AA)
    # # A = AAm.array()
    # A = AAm

    # # block diagonal preconditioner
    # P_up = inner(up, vp) * dxDarcy + inner(div(up), div(vp)) * dxDarcy # Hdiv
    # P_pp = inner(pp, wp) * dxDarcy # L2
    # P_dp = (inner(dp, ep) + inner(grad(dp), grad(ep))) * dxDarcy # H1
    # P_uf = (inner(uf, vf) + inner(grad(uf), grad(vf))) * dxStokes # H1
    # P_pf = (inner(pf, wf)) * dxStokes # H1
    # P_up, P_pp, P_dp, P_uf, P_pf = map(
    #     lambda form: ii_convert(ii_assemble(form)),
    #     [P_up, P_pp, P_dp, P_uf, P_pf]
    # )1

    # P_lbd = hsmg.HsNorm(W[-1], s=0.5)
    # P_lbd * Function(W[-1]).vector() # enforces lazy computation

    # P_lbd = P_lbd.matrix

    # P = ii_convert(
    #     block_diag_mat([P_up, P_pp, P_dp, P_uf, P_pf, P_lbd]), ""
    # )

    # # TODO: should I use the bbcs object returned from the call to bbcs.apply() above here?
    # # or would that only be for if i wanted to apply to vectors?
    # bbcs.apply(
    #     P
    # )
    # # P = ii_convert(P).array()
    # P = ii_convert(P).array()
    return A, P

def condition_number(A, P):
    eigs = scipy.linalg.eigh(A, P, eigvals_only=True)
    eigs = sorted(map(abs, eigs))
    return max(eigs)/min(eigs)


def vary_param_experiment():
    def vary_param(par_to_vary, par_min, par_max, output_fn, N=4, N_pars=11):
        params = dict(brain_params)
        pars = np.exp(np.linspace(np.log(par_min), np.log(par_max), N_pars))
        results = {}


        for par in pars:
            params[par_to_vary] = par
            domain = AmbartsumyanMMSDomain(N)

            A, P = compute_A_P_BDS(domain, params)

            eigs = scipy.linalg.eigh(A, P, eigvals_only=True)
            eigs = sorted(map(abs, eigs))

            condition_number = max(eigs)/min(eigs)

            results[par] = condition_number

        x = sorted(results.keys())
        y = [results[xi] for xi in x]

        plt.loglog(x, y, "ro-")
        plt.xlabel(par_to_vary)
        plt.ylabel("K(P*A)")
        plt.title("Darcy-Biot-Stokes (N={})".format(N))
        plt.savefig(output_fn)
        plt.close()

        return results
    output_dir = os.path.join("output", "param_condition_numbers")

    condition_numbers = {}


    for par_to_vary, par_min, par_max in [
            (parname, brain_params[parname]*1E-2, brain_params[parname]*1E2)

            # for parname in ["K"]

            for parname in brain_params
            if parname not in ["Cp"]
    ]:
        output_fn = os.path.join(output_dir, "{}.png".format(par_to_vary))
        condition_numbers[par_to_vary] = vary_param(par_to_vary, par_min, par_max, output_fn, N=8, N_pars=13)



    for parname in condition_numbers:
        par_condition_numbers = condition_numbers[parname]
        parmin, parmax = min(par_condition_numbers.keys()), max(par_condition_numbers.keys())
        Kmin, Kmax = par_condition_numbers[parmin], par_condition_numbers[parmax]
        print ("\n{parname: <10}\n  {parname: >10}={parmin:.4e}, K={Kmin:.4e}\n"
               "  {parname: >10}={parmax:.4e}, K={Kmax:.4e}").format(
                   parname=parname, parmin=parmin, parmax=parmax, Kmin=Kmin, Kmax=Kmax
    )


    def test(*args, **kwargs):
        print args
        print kwargs

    with open(os.path.join(output_dir, "par_condition_numbers.pickle"), "w") as f:
        pickle.dump(condition_numbers, f)



    def pandas_latex_cleaner(s):
        s = s.replace("\_", "_")
        s = s.replace("\\textbackslash", "\\")

        for parname in sorted(condition_numbers.keys()): # sorted is hack to keep alpha_BJS after alpha
            if parname == "K":
                parname = "kappa"
            s = s.replace(parname, "${}$".format(parname))

        s = s.replace("lbd", "\\lambda")
        s = s.replace("alpha", "\\alpha")
        s = s.replace("kappa", "\\kappa")
        s = s.replace("mu", "\\mu")
        s = s.replace("s0", "s_0")
        s = s.replace("$_BJS", "_{\\text{BJS}}$")
        return s

    def exp_formatter(x):
        from math import log, floor
        if x == 0:
            mantissa = 0
            exponent = 0
        else:
            exponent = int(floor(log(abs(x))/log(10)))
            mantissa = x/10**exponent
        return "{:.4f}\:E{}".format(mantissa, exponent)

    def param_dict_to_latex(params):
        return pandas_latex_cleaner(pd.DataFrame({"val": params, "source": "", "units": ""}).to_latex(float_format=exp_formatter))

    def condition_numbers_to_latex_table(condition_numbers):
        d = {}

        for parname in condition_numbers:
            res = condition_numbers[parname]
            if parname == "K":
                parname = "kappa"

            pmin = min(res.keys())
            pmid = sorted(res.keys())[len(res.keys())/2]
            pmax = max(res.keys())
            K0 = res[pmid]
            d[parname] = {
                "Kmin": res[pmin],
                # "Kmid": res[pmid],
                "Kmax": res[pmax]
            }
        df = pd.DataFrame(d).transpose()

        df /= K0

        s = df.to_latex(float_format=exp_formatter)

        return "K0=" + exp_formatter(K0) + "\n" + pandas_latex_cleaner(s)
    print("\n" + condition_numbers_to_latex_table(condition_numbers))
    print("\n" + param_dict_to_latex(brain_params))




# run_experiment()
