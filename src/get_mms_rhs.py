from sympy import diff, symbols, pi, cos, sin, exp, sqrt

x, y, t = symbols("x y t")


def laplace(u):
    """Given a sympy function of x, y, returns their laplacian"""
    return diff(u, x, x) + diff(u, y, y)


def vector_laplace(u):
    """Given a SympyVector, returns a SympyVector of componentwise laplacian"""

    return SympyVector(
        *[diff(ui, x, x) + diff(ui, y, y) for ui in u]
    )


def grad(u):
    """Given a sympy expression, return its gradient as a SympyVector"""
    return SympyVector(diff(u, x), diff(u, y))


def div(u):
    u1, u2 = u.x, u.y
    return diff(u1, x) + diff(u2, y)


def div_sym_grad(u):
    """Given a SympyVector u, returns 0.5*div(grad(u)+grad(u)T)"""
    assert len(u) == 2
    return SympyVector(
        diff(u.x, x, x) + (diff(u.x, y, y) + diff(u.y, y, x)) / 2,
        diff(u.y, x, x) + (diff(u.y, y, y) + diff(u.x, y, x)) / 2,
    )


def inner(u, v):
    """Given 2 SympyVectors, computes their scalar product."""
    assert len(u) == len(v)
    return sum(ui * vi for ui, vi in zip(u, v))


class SympyVector(object):
    """Wrapper around a list of sympy exprs for doing componentwise addition"""

    def __init__(self, *args):
        assert (len(args) == 2 or len(args) == 3)
        self._l = args

    def __add__(self, other):
        return SympyVector(
            *[si + oi for si, oi in zip(self, other)]
        )

    def __sub__(self, other):
        return self + (-1) * other

    def __iter__(self):
        return self._l.__iter__()    # copy to prevent modification of list

    def __repr__(self):
        try:
            u = self.simplify()
        except:
            u = self

        strings = map(str, u)
        N = max(map(len, strings))

        return "".join(
            "|  {}  |\n".format(si.center(N))
            for si in strings
        )

    def __len__(self):
        return len(self._l)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        try:                    # checks that you are multiplying by a scalar
            # otherwise, vector*vector works but isn't what you expect
            len(other)
        except TypeError:
            pass
        return SympyVector(
            *[other * si for si in self]
        )

    @property
    def x(self):
        return self._l[0]

    @property
    def y(self):
        return self._l[1]

    @property
    def z(self):
        return self._l[2]

    def simplify(self):
        return SympyVector(*[vi.simplify() for vi in self])


# problem-specific stuff
# parameters
mu_f, mu_p, lbd_p, eta_p, alpha_BJS, alpha, K, s0, DP = symbols(
    "mu_f mu_p lbd_p eta_p alpha_BJS alpha K s0 DP"
)

mu_f, mu_p, lbd_p, eta_p, alpha_BJS, alpha, K, s0, DP = [1] * 9

# DP=0


# uf, up, dp: SympyVectors, pf, pp: sympy expressions

def norm_sym_grad(u, n):
    """Given SympyVectors u, n, computes D(u) * n, where * is matrix/vector multiplication"""
    Du_x = SympyVector(diff(u.x, x), (diff(u.x, y) + diff(u.y, x)) / 2)
    Du_y = SympyVector((diff(u.x, y) + diff(u.y, x)) / 2, diff(u.y, y))
    return SympyVector(inner(Du_x, n), inner(Du_y, n))


def verify_BJS(up, pp, dp, uf, pf):
    assert False


def stokes_RHS_ff(uf, pf):
    return grad(pf) - 2 * mu_f * div_sym_grad(uf)


def stokes_RHS_qf(uf):
    return div(uf)


def biot_RHS_fp(dp, pp):
    return grad(alpha * pp - lbd_p * div(dp)) - 2 * mu_p * div_sym_grad(dp)


def biot_RHS_gp(up, pp):        # not in paper, but diff. between up and -grad(pp)
    return grad(pp) + mu_p / K * up


def biot_RHS_qp(dp, pp, up):
    ddt = diff(s0 * pp + alpha * div(dp), t)
    return ddt + div(up)


def ambartsumyan_mms_solution():
    up = pi * exp(t) * SympyVector(-cos(pi * x) * cos(pi * y / 2),
                                   sin(pi * x) * sin(pi * y / 2) / 2)
    pp = exp(t) * sin(pi * x) * cos(pi * y / 2)
    dp = sin(pi * t) * SympyVector(-3 * x + cos(y), y + 1)

    uf = pi * cos(pi * t) * SympyVector(-3 * x + cos(y), y + 1)
    pf = exp(t) * sin(pi * x) * cos(pi * y / 2) + 2 * pi * cos(pi * t) + DP
    return up, pp, dp, uf, pf


def verify_interface_conditions(up, pp, dp, uf, pf):
    for v in (up, uf, dp):
        assert isinstance(v, SympyVector)

    for s in (pp, pf):
        assert not isinstance(s, SympyVector)

    def restrict(expr):
        """Restrict expr to the interface y=0."""
        return expr.subs(y, 0)

    nf = SympyVector(0, -1)
    np = -1 * nf
    tau = SympyVector(1, 0)

    # sigma_f n_f * nf
    normstress_f = 2 * mu_f * norm_sym_grad(uf, nf) - pf * nf
    normstress_p = lbd_p * div(dp) * np + 2 * mu_p * \
        norm_sym_grad(dp, np) - alpha * pp * np
    ddpdt = SympyVector(diff(dp.x, t), diff(dp.y, t))

    # 2.6: mass conservation (subs to be on interface y=0)
    mass_conservation = restrict(inner(uf, nf) + inner(ddpdt + up, np))
    assert mass_conservation == 0

    # 2.7: stress balance
    stressbalance_1 = restrict(inner(normstress_f, nf) + pp + DP)
    assert stressbalance_1 == 0

    stressbalance_2 = normstress_f + normstress_p + nf * DP
    for i, component in enumerate(stressbalance_2):
        assert restrict(component) == 0

    # 2.8: BJS
    shearstress = restrict(
        inner(normstress_f, tau)
        + mu_f * alpha_BJS / sqrt(K) * inner(uf - ddpdt, tau)
    )
    assert shearstress == 0


def print_all_RHSes(up, pp, dp, uf, pf):
    # verify that up = -mu_p/K * grad(pp)
    # this doesn't really belong here, but if it's not i guess a RHS term would be needed so kinda?
    gp = biot_RHS_gp(up, pp)
    for component in (gp.x, gp.y):
        assert component.subs([(mu_p, -2), (K, -2)]).simplify() == 0

    print "Stokes:"
    print "ff: \n", stokes_RHS_ff(uf, pf)
    print "qf: \n", stokes_RHS_qf(uf)

    print "\nBiot:"
    print "fp: \n", biot_RHS_fp(uf, pf)
    print "qp: \n", biot_RHS_qp(dp, pp, up)


mms_sol = ambartsumyan_mms_solution()
verify_interface_conditions(*mms_sol)
print_all_RHSes(*mms_sol)

up, pp, dp, uf, pf = mms_sol
