import dolfinx
import basix

from . import _scifem


def create_real_functionspace(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.FunctionSpace:
    ufl_e = basix.ufl.element("P", mesh.basix_cell(), 0, dtype=float, discontinuous=True)
    cppV = _scifem.create_real_functionspace(mesh._cpp_object)
    return dolfinx.fem.FunctionSpace(mesh, ufl_e, cppV)
