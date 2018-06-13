from precond_experiment_utils import *
from dolfin import *
from block import *
from xii import *

from numpy.linalg import inv

brain_params = {
    "alpha": 0.9995,
    "lbd_p": 7142,
    "mu_p": 1786,
    "K": 1E-5,
    "dt": 1E-3,
    # "s0": 1/lbd_p
}

brain_params["s0"] = 1/float(brain_params["lbd_p"])

for k in brain_params:
    brain_params[k] = Constant(brain_params[k])

dt = brain_params["dt"]
alpha = brain_params["alpha"]
s0 = brain_params["s0"]
mu_p = brain_params["mu_p"]
lbd_p = brain_params["lbd_p"]
K = brain_params["K"]



# domain
N = 4
mesh = UnitSquareMesh(N, N)
dx = Measure("dx", domain=mesh)

# function spaces
Vp = FunctionSpace(mesh, "RT", 2)
Qp = FunctionSpace(mesh, "DG", 1)
U = VectorFunctionSpace(mesh, "CG", 2)
W = [Vp, Qp, U]

up, pp, dp = map(TrialFunction, W)
vp, wp, ep = map(TestFunction, W)


# system matrix
aep = (1 / dt) * (
    (mu_p) * inner(sym(grad(dp)), sym(grad(ep))) * dx
    + (lbd_p) * inner(div(dp), div(ep)) * dx
)


a = [
    [(mu_p / K) * inner(up, vp) * dx,    -div(vp) * pp * dx,                                 0],
    [-div(up) * wp * dx,          -(s0 / dt) * pp * wp * dx, (alpha / dt) * -div(dp) * wp * dx],
    [0,                   (alpha / dt) * -div(ep) * pp * dx,                               aep],
]

# preconditioner
p_diag =[
    a[0][0],
    (
        ((s0/dt) + (1/(mu_p + lbd_p))) * pp * wp * dx
        + (K/mu_p) * inner(grad(pp), grad(wp)) * dx
    ),
    a[2][2]
]
p = [[p_diag[i] if i==j else 0 for j in range(3)] for i in range(3)]


# BCs
bcs = [
    [DirichletBC(W[0], ((0, 0)), "on_boundary")], # up
    [],                                                   # pp
    [DirichletBC(W[2], ((0, 0)), "on_boundary")], # dp
]
bbcs = block_bc(bcs, symmetric=True)


AA, PP = (ii_convert(ii_assemble(bl), "") for bl in (a, p))
for MM in (AA, PP):
    bbcs.apply(
        MM
    )
AA, PP = to_numpy(AA, block=False), to_numpy(PP, block=False)

# for some reason, replacing A by P.inv * A and P by I makes condition number huge
# AA, PP = inv(PP)*AA, np.eye(AA.shape[0])

eigs = scipy.linalg.eigh(AA, PP, eigvals_only=True)
eigs = sorted(map(abs, eigs))
print eigs[-1]/eigs[0]



