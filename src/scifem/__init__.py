from mpi4py import MPI
import dolfinx
import basix
import numpy as np
import ufl
from . import _scifem  # type: ignore

__all__ = ["create_real_functionspace", "assemble_scalar"]


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


def assemble_scalar(J: ufl.form.Form | dolfinx.fem.Form) -> np.floating | np.complexfloating:
    """Assemble a scalar form and gather result across processes

    Args:
        form: The form to assemble.

    Returns:
        The accumulated value of the assembled form.
    """
    compiled_form = dolfinx.fem.form(J)
    if (rank := compiled_form.rank) != 0:
        raise ValueError(f"Form must be a scalar form, got for of arity {rank}")
    local_result = dolfinx.fem.assemble_scalar(compiled_form)
    return compiled_form.mesh.comm.allreduce(local_result, op=MPI.SUM)
