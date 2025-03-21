from __future__ import annotations

from mpi4py import MPI
import ufl
import numpy as np
import dolfinx
from typing import Literal
import numpy.typing as npt
import typing


def _extract_dtype(expr: ufl.form.Form | ufl.core.expr.Expr | dolfinx.fem.Form) -> npt.DTypeLike:
    """Given a ufl form, expression or a compiled DOLFINx form extract the data type that should
    be used in form compilation.

    Args:
        form: The form to extract the data type from.

    """
    try:
        return expr.dtype
    except AttributeError:
        try:
            geom_type = ufl.domain.extract_domains(expr)[0].ufl_cargo().geometry.x.dtype
        except AttributeError:
            # Legacy support
            geom_type = expr.ufl_domain().ufl_cargo().geometry.x.dtype
        coefficients = expr.coefficients()
        constants = expr.constants()
        if (not coefficients) and (not constants):
            return geom_type
        else:
            data_types: set[npt.DTypeLike] = set()
            for coefficient in coefficients:
                data_types.add(coefficient.dtype)
            for constant in constants:
                data_types.add(constant.dtype)

            if len(data_types) > 1:
                raise RuntimeError("All coefficients and constants must have the same data type")
            assert len(data_types) == 1
            return data_types.pop()


def assemble_scalar(
    J: ufl.form.Form | dolfinx.fem.Form,
    entity_maps: typing.Optional[dict[dolfinx.mesh.Mesh, npt.NDArray[np.int32]]] = None,
) -> np.floating | np.complexfloating:
    """Assemble a scalar form and gather result across processes

    Args:
        form: The form to assemble.
        entity_maps: Maps of entities on related submeshes to the domain used in `J`.

    Returns:
        The accumulated value of the assembled form.
    """
    dtype = _extract_dtype(J)
    compiled_form = dolfinx.fem.form(J, entity_maps=entity_maps, dtype=dtype)

    if (rank := compiled_form.rank) != 0:
        raise ValueError(f"Form must be a scalar form, got for of arity {rank}")
    local_result = dolfinx.fem.assemble_scalar(compiled_form)
    return compiled_form.mesh.comm.allreduce(local_result, op=MPI.SUM)


def norm(
    expr: ufl.core.expr.Expr,
    norm_type: Literal["L2", "H1", "H10"],
    entity_maps: typing.Optional[dict[dolfinx.mesh.Mesh, npt.NDArray[np.int32]]] = None,
) -> dolfinx.fem.Form:
    """
    Compile the norm of an UFL expression into a DOLFINx form.

    Args:
        expr: UFL expression
        norm_type: Type of norm
        entity_maps: Mapping for Constants and Coefficients within the expression that
            lives on a submesh.
    """
    if norm_type == "L2":
        form = ufl.inner(expr, expr) * ufl.dx
    elif norm_type == "H1":
        form = ufl.inner(expr, expr) * ufl.dx + ufl.inner(ufl.grad(expr), ufl.grad(expr)) * ufl.dx
    elif norm_type == "H10":
        form = ufl.inner(ufl.grad(expr), ufl.grad(expr)) * ufl.dx
    else:
        raise RuntimeError(f"Unexpected norm type: {norm_type}")
    dtype = _extract_dtype(form)
    return dolfinx.fem.form(form, entity_maps=entity_maps, dtype=dtype)
