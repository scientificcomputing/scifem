import dolfinx
import basix
import numpy as np
from . import _scifem


def create_real_functionspace(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.FunctionSpace:

    
    ufl_e = basix.ufl.element("P", mesh.basix_cell(), 0, dtype=float, discontinuous=True)
    
    if (dtype:=mesh.geometry.x.dtype) == np.float64:
    
        cppV = _scifem.create_real_functionspace_float64(mesh._cpp_object)
    elif dtype == np.float32:
        cppV = _scifem.create_real_functionspace_float32(mesh._cpp_object)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dolfinx.fem.FunctionSpace(mesh, ufl_e, cppV)
