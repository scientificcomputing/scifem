import dolfinx
import basix
import numpy as np
from . import _scifem  # type: ignore


def create_real_functionspace(
    mesh: dolfinx.mesh.Mesh, value_shape: tuple[int, ...] = ()
) -> dolfinx.fem.FunctionSpace:
    """Create a real function space.

    Args:
        mesh: The mesh the real space is defined on.
        value_shape: The shape of the values in the real space.

    Returns:
        The real valued function space.
    Note:
        For scalar elements value shape is ``()``.

    """

    dtype = mesh.geometry.x.dtype
    ufl_e = basix.ufl.element(
        "P", mesh.basix_cell(), 0, dtype=dtype, discontinuous=True, shape=value_shape
    )

    if (dtype := mesh.geometry.x.dtype) == np.float64:
        cppV = _scifem.create_real_functionspace_float64(mesh._cpp_object, value_shape)
    elif dtype == np.float32:
        cppV = _scifem.create_real_functionspace_float32(mesh._cpp_object, value_shape)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dolfinx.fem.FunctionSpace(mesh, ufl_e, cppV)
