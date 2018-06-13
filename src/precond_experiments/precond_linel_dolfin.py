from precond_experiment_utils import *
from dolfin import *
from block import *
from xii import *

from numpy.linalg import inv

brain_params = {
    "mu": 1786,
    "lbd": 7142000,
}

for k in brain_params:
    print k, brain_params[k]
    brain_params[k] = Constant(brain_params[k])

mu = brain_params["mu"]
lbd = brain_params["lbd"]


# domain
N = 4
mesh = UnitSquareMesh(N, N)
dx = Measure("dx", domain=mesh)

# function spaces
U = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = [U, Q]

dp, pp = map(TrialFunction, W)
ep, wp = map(TestFunction, W)


# system matrix

a = [
    [mu * inner(grad(dp), grad(ep)) * dx,   inner(div(ep), pp)*dx],
    [          inner(div(dp), wp)*dx,       -1/lbd * pp * wp * dx]
]

p = [
    [mu * inner(grad(dp), grad(ep)) * dx,                 0],
    [          0,                       1/mu * pp * wp * dx]
]


# BCs
bcs = [
    [DirichletBC(W[0], ((0, 0)), "on_boundary")], # dp
    [DirichletBC(W[1], 0, "on_boundary")], # pp
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



