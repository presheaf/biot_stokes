import numpy as np
from numpy.linalg import inv
from numpy import zeros
import scipy.linalg

# small script demonstrating(?) that systems on the form
# [[ A0   B.T    ],
#  [  B   -A1  C.T],
#  [       C   A2]]
# 
# with A0, A1, A2 SPD
# can be preconditioned by
# 
# [[ A0,                  ],
#  [     A1 + S0 + S2     ],
#  [                   A2 ]]
#
# where S0 = B*inv(A0)*B.T, S2 = C.T*inv(A2)*C
# I obtained this by 'row'-reducing AA. 

C = 5                          # increase this to see that condition number is unchanged

np.random.seed(42)
Ns = C*np.array([37, 12, 42])   # no good reason for these not being equal except that it's easier to catch mistakes
N0, N1, N2 = Ns


def random_mat(a, b, spd=False):
    M = np.random.rand(a, b)
    if spd:
        assert a == b
        return np.matrix(np.matmul(M.T, M))
    else:
        return np.matrix(M)

def dezero(MM):                 # replaces None-s with np.zeros() of apporpriate size
    N = len(MM)
    for i in range(N):
        for j in range(N):
            if MM[i][j] is None:
                MM[i][j] = np.zeros((Ns[i], Ns[j]))
    return MM

mup = 1E3
lbdp = 1E4
K = 1E-5
dt = 1E-3
alpha = 1
s0 = 1E-4

A0 = mup/K * random_mat(N0, N0, spd=True)
A1 = s0/dt * random_mat(N1, N1, spd=True)

A21 = (mup/dt) * random_mat(N2, N2, spd=True)
A22 = (lbdp/dt) * random_mat(N2, N2, spd=True)
A2 = A21 + A22
B = random_mat(N1, N0)
C = alpha/dt * random_mat(N2, N1)

AA = np.block(dezero(
    [[A0, B.T, None],
     [B, -A1, C.T],
     [None, C, A2]]
))

PP = np.block(dezero(
    [[A0, None, None],
     [None, A1 + (B*inv(A0)*B.T) + (C.T*inv(A2)*C), None],
     [None, None, A2]]
))

        

eigs = scipy.linalg.eigh(AA, PP, eigvals_only=True)
eigs = sorted(map(abs, eigs))
print "Condition number: {:.6e}".format(max(eigs)/min(eigs))




