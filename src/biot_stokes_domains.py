import mshr
import dolfin
from block import block_mat, block_vec, block_bc
# from petsc4py import PETSc
from dolfin import *
from xii import *
import itertools
import hsmg


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


class Tunnel2DDomain(object):
    def __init__(self, N):

        _mesh = RectangleMesh(Point(0, -1), Point(1, 1), N, 2 * N)

        stokes_subdomain = dolfin.CompiledSubDomain("(3*x[1] < 1) || (3*x[1] > -1)")

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
        CompiledSubDomain(
            "on_boundary && (near(3*x[1], 1) && near(3*x[1], -1))"
        ).mark(surfaces, 1)
        self.interface = EmbeddedMesh(surfaces, 1)

        self.mark_boundary()

    @property
    def dimension(self):
        return 2

    @property                   # interface normal pointing outwards from fluid domain
    def interface_normal_f(self):
        return OuterNormal(self.interface, [0.0] * self.dimension)
    @property
    def interface_normal_p(self):
        return OuterNormal(self.interface, [-0.0] * self.dimension)


    def mark_boundary(self):
        """Interface should be marked as 1. Do not set BCs there.
        Other bdy is 2"""

        stokes_markers = MeshFunction("size_t", self.stokes_domain, 1, 0)
        porous_markers = MeshFunction("size_t", self.porous_domain, 1, 0)

        interface_bdy = dolfin.CompiledSubDomain(
            "(near(3*x[1], 1) || near(3*x[1], -1)) && on_boundary")
        other_bdy = dolfin.CompiledSubDomain("on_boundary")

        for markers in [stokes_markers, porous_markers]:
            other_bdy.mark(markers, 2)
            interface_bdy.mark(markers, 1)

        self.stokes_bdy_markers = stokes_markers
        self.porous_bdy_markers = porous_markers
