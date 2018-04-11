from sympy import diff, symbols, pi, cos, sin, exp

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
        return self + (-1)*other

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

## problem-specific stuff
# parameters
mu_f, mu_p, lbd_p, eta_p, alpha_BJS, alpha, K, s0= symbols(
    "mu_f mu_p lbd_p eta_p alpha_BJS alpha K s0"
)

# uf, up, dp: SympyVectors, pf, pp: sympy expressions

def verify_BJS(up, pp, dp, uf, pf):
    assert False


def stokes_RHS_ff(uf, pf):
    return grad(pf) - 2 * mu_f * div_sym_grad(uf)


def stokes_RHS_qf(uf):
    return div(uf)


def biot_RHS_fp(dp, pp):
    return grad(alpha * pp - lbd_p * div(dp)) - 2 * mu_p * div_sym_grad(dp)

def biot_RHS_gp(up, pp):        # not in paper, but diff. between up and -grad(pp)
    return grad(pp) + mu_p/K * up

def biot_RHS_qp(dp, pp, up):
    ddt = diff(s0*pp + alpha*div(dp), t)
    return ddt + div(up)


def print_all_RHSes():
    pf = exp(t) * sin(pi * x) * cos(pi * y / 2) + 2 * pi * cos(pi * t)
    uf = pi * cos(pi * t) * SympyVector(-3 * x + cos(y), y + 1)

    pp = exp(t) * sin(pi * x) * cos(pi * y / 2)
    up = pi * exp(t) * SympyVector(cos(pi * x) * cos(pi * y / 2),
                                   1 / 2 * sin(pi * x) * sin(pi * y / 2))
    dp = sin(pi * t) * SympyVector(-3 * x + cos(y), y + 1)

    for v in (up, uf, dp):
        assert isinstance(v, SympyVector)
        
    for s in (pp, pf):
        assert not isinstance(s, SympyVector)

    print "Stokes:"
    print "ff: \n", stokes_RHS_ff(uf, pf)
    print "qf: \n", stokes_RHS_qf(uf)

    print "\nBiot:"
    print "fp: \n", biot_RHS_fp(uf, pf)
    print "gp: \n", biot_RHS_gp(up, pp)
    print "qp: \n", biot_RHS_qp(dp, pp, up)


print_all_RHSes()
