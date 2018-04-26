from sympy import diff, symbols, pi, cos, sin, exp, sqrt
import sympy

x, y, t = symbols("x[0] x[1] t")


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

def vector_grad(u):
    """Given a SympyVector, return its gradient as a SympyMatrix"""
    return SympyMatrix(
        [
            [diff(u.x, x), diff(u.x, y)],
            [diff(u.y, x), diff(u.y, y)],
        ]
    )

def div(u):
    return diff(u.x, x) + diff(u.y, y)


def div_sym_grad(u):
    """Given a SympyVector u, returns 0.5*div(grad(u)+grad(u)T)"""
    assert len(u) == 2
    return SympyVector(
        diff(u.x, x, x) + (diff(u.x, y, y) + diff(u.y, y, x)) / 2,
        diff(u.y, y, y) + (diff(u.y, x, x) + diff(u.x, y, x)) / 2,
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

    def __neg__(self):
        return (-1) * self

    def __iter__(self):
        return self._l.__iter__()    # copy to prevent modification of list

    def __repr__(self):
        try:
            u = self.simplify()
        except:
            u = self

        strings = map(sympy.printing.ccode, u)
        # N = max(map(len, strings))

        return "(\n  {}\n)".format(",\n  ".join(
            "\"{}\"".format(si)
            for si in strings
        )).replace(".0L", ".0").replace("M_PI", "pi")

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


class SympyMatrix(object):
    """Wrapper around a list of sympy exprs for doing componentwise addition"""

    def __init__(self, rows, flat=False):
        """Stores internally as flat list. If flat=True, treat rows as 'list sum' of rows."""
        if not flat:
            assert (len(rows) == 2 or len(rows) == 3)
            for row in rows:
                assert len(row) == len(rows)
            self._l = sum(rows, [])
        else:
            assert len(rows) == 4 or len(rows) == 9
            self._l = rows

    def __add__(self, other):
        return SympyMatrix(
            *[si + oi for si, oi in zip(self, other)],
            flat=True
        )

    def __sub__(self, other):
        return self + (-1) * other

    def __neg__(self):
        return (-1) * self

    def __iter__(self):
        return self._l.__iter__()    # copy to prevent modification of list

    def __repr__(self):
        try:
            u = self.simplify()
        except:
            u = self

        N = len(self)
        strings = [["\"{}\"".format(sympy.printing.ccode(self[i,j])) for j in range(N)] for i in range(N)]
        
        L = max(map(lambda r: max(map(len, r)), strings))

        rows = ["({})".format(",  ".join(map(lambda s: s.center(L), row))) for row in strings]
        
        

        return "(\n  {}\n)".format(",\n  ".join(
            "{}".format(ri)
            for ri in rows
        )).replace(".0L", ".0").replace("M_PI", "pi")

    def __len__(self):          # dimension, not number of elts
        if len(self._l) == 4:
            return 2
        elif len(self._l) == 9:
            return 3
        raise Exception

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        try:                    # checks that you are multiplying by a scalar
            # otherwise, vector*vector works but isn't what you expect
            len(other)
        except TypeError:
            pass
        return SympyMatrix(
            *[other * si for si in self],
            flat=True
        )

    def __getitem__(self, key):
        assert len(key) == 2
        i, j = key
        N = len(self)
        return self._l[N*i + j]

    def simplify(self):
        return SympyMatrix(*[vi.simplify() for vi in self])


# problem-specific stuff
# parameters
mu_f, mu_p, lbd_p, eta_p, alpha_BJS, alpha, K, s0, DP = symbols(
    "mu_f mu_p lbd_p eta_p alpha_BJS alpha K s0 DP"
)

mu_f, mu_p, lbd_p, eta_p, alpha_BJS, alpha, K, s0, DP = [1] * 9


alpha_BJS = 0
DP= 0
dt = 1E-4
# uf, up, dp: SympyVectors, pf, pp: sympy expressions

## stokes/darcy/biot
def norm_sym_grad(u, n):
    """Given SympyVectors u, n, computes D(u) * n, where * is matrix/vector multiplication"""
    Du_x = SympyVector(diff(u.x, x), (diff(u.x, y) + diff(u.y, x)) / 2)
    Du_y = SympyVector((diff(u.x, y) + diff(u.y, x)) / 2, diff(u.y, y))
    return SympyVector(inner(Du_x, n), inner(Du_y, n))


def stokes_RHS_ff(uf, pf):
    return grad(pf) - 2 * mu_f * div_sym_grad(uf)


def stokes_RHS_qf(uf):
    return div(uf)


def biot_RHS_fp(dp, pp):
    return grad(alpha * pp - lbd_p * div(dp)) - 2 * mu_p * div_sym_grad(dp)


def biot_RHS_gp(up, pp):        # not in paper, but diff. between up and -grad(pp)
    return K / mu_p * grad(pp) + up


def BE_dt(expr, dt):
    """Compute discrete time derivative of expr using BE discretization."""
    # return (expr - expr.subs(t, t-dt))/dt; 
    return diff(expr, t)

def biot_RHS_qp(dp, pp, up):
    # ddt = diff(s0 * pp + alpha * div(dp), t)
    ddt = BE_dt(s0 * pp + alpha * div(dp), dt)
    return ddt + div(up)

# ## Poisson
# def stokes_poisson_RHS_ff(uf):
#     return uf-vector_laplace(uf)

# def stokes_poisson_RHS_qf(pf):
#     return pf-laplace(pf)

# def biot_poisson_RHS_fp(dp):


# def darcy_poisson_RHS_gp(up):
#     return -vector_laplace(up)

# def biot_poisson_RHS_qp(pp):
#     return -laplace(pp)


def ambartsumyan_mms_solution():
    up = pi * exp(t) * SympyVector(-cos(pi * x) * cos(pi * y / 2),
                                   sin(pi * x) * sin(pi * y / 2) / 2)
    pp = exp(t) * sin(pi * x) * cos(pi * y / 2)
    dp = sin(pi * t) * SympyVector(-3 * x + cos(y), y + 1)

    uf = pi * cos(pi * t) * SympyVector(-3 * x + cos(y), y + 1)
    pf = exp(t) * sin(pi * x) * cos(pi * y / 2) + 2 * pi * cos(pi * t) + DP
    return up, pp, dp, uf, pf

def simple_mms_solution():
    up = SympyVector(x**2 * y, x*y)
    pp = y**3
    dp = SympyVector(y, y)
    uf = SympyVector(2*y, y*x**4)
    pf = y*(-1 + 2*x)
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
    normstress_p = (
        lbd_p * div(dp) * np +
        2 * mu_p * norm_sym_grad(dp, np)
        - alpha * pp * np
    )
    # # ddpdt = SympyVector(diff(dp.x, t), diff(dp.y, t))
    # ddpdt = SympyVector(BE_dt(dp.x, t), BE_dt(dp.y, t))
    
    # # 2.6: mass conservation (subs to be on interface y=0)
    # mass_conservation = restrict(inner(uf, nf) + inner(ddpdt + up, np))
    # assert mass_conservation == 0

    # # 2.7: stress balance
    # stressbalance_1 = restrict(inner(normstress_f, nf) + pp + DP)
    # assert stressbalance_1 == 0

    # stressbalance_2 = normstress_f + normstress_p + nf * DP
    # for i, component in enumerate(stressbalance_2):
    #     assert restrict(component) == 0

    # # 2.8: BJS
    # shearstress = restrict(
    #     inner(normstress_f, tau)
    #     + mu_f * alpha_BJS / sqrt(K) * inner(uf - ddpdt, tau)
    # )
    # assert shearstress == 0


def print_all_RHSes(up, pp, dp, uf, pf):

    def exprify(u):
        
        if isinstance(u, SympyVector):
            s = "Expression(\n{}, degree=5, t=0\n)".format(u)
        else:
            s= "Expression(\n\"{}\", degree=5, t=0\n)".format(
                sympy.printing.ccode(u)
            )

        return s.replace(".0L", ".0").replace("M_PI", "pi")
        
    
    print 
    print "s_vp =", exprify(biot_RHS_gp(up, pp))
    print "s_wp =", exprify(biot_RHS_qp(dp, pp, up))
    print "s_ep = ", exprify(biot_RHS_fp(dp, pp))
    print "s_vf =", exprify(stokes_RHS_ff(uf, pf))
    print "s_wf = ", exprify(stokes_RHS_qf(uf))

    
    

mms_sol = ambartsumyan_mms_solution()
# mms_sol = simple_mms_solution()
verify_interface_conditions(*mms_sol)
print_all_RHSes(*mms_sol)

# up, pp, dp, uf, pf = mms_sol


def print_all_exact_solutions(up, pp, dp, uf, pf):
    for name, func in zip(
            ["up", "pp", "dp", "uf", "pf"],
            [up, pp, dp, uf, pf]
    ):
        if isinstance(func, SympyVector):
            s = str(func)
        else:
            s = "\"{}\"".format(sympy.printing.ccode(func))
        print "{}_e=Expression(\n{}, degree=5, t=0\n)".format(
            name, s
        ).replace(".0L", ".0").replace("M_PI", "pi")

def print_all_gradients(up, pp, dp, uf, pf):
    print "\n------------------------------\n\nGradient:"
    for func, name in zip(
            [up, pp, dp, uf, pf],
            ["up", "pp", "dp", "uf", "pf"],
            
    ):
        if isinstance(func, SympyVector): # TODO: matrix stuff happens here, spin it out into SympyMatrix
            name_x, name_y = name + "_x: \n", name + "_y: \n"
            u1, u2 = func.x, func.y
            u1x, u1y, u2x, u2y = diff(u1, x), diff(u1, y), diff(u2, x), diff(u2, y)
            grad_string = str(SympyMatrix([[u1x, u1y], [u2x, u2y]]))
            # grad_string = "((\"{}\", \"{}\"),\n(\"{}\", \"{}\")),\n".format(
            #     *map(str, (u1x, u2x, u1y, u2y))
            # )
        else:
            grad_string = str(grad(func))

        print "g_{}=Expression(\n{}, degree=3, t=0\n)".format(name, grad_string)


print ""
print_all_exact_solutions(*mms_sol)
# print_all_gradients(*mms_sol)
