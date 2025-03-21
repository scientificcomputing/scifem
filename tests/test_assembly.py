from mpi4py import MPI

from scifem import assemble_scalar, norm
from dolfinx.mesh import create_unit_square
from dolfinx.fem import Function, functionspace, Constant
import ufl
import pytest
import numpy as np


@pytest.mark.parametrize("gtype", [np.float64, np.float32])
def test_assemble_scalar_spatial(gtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=gtype)
    x = ufl.SpatialCoordinate(mesh)
    f = x[0] ** 2 * x[1] * ufl.dx
    tol = 50 * np.finfo(gtype).eps
    assert np.isclose(assemble_scalar(f), 1 / 3 * 1 / 2, atol=tol)


@pytest.mark.parametrize(
    "gtype, dtype",
    [
        [np.float64, np.float64],
        [np.float64, np.complex128],
        [np.float32, np.float32],
        [np.float32, np.complex64],
    ],
)
def test_assemble_scalar_constant(gtype, dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=gtype)
    f = Constant(mesh, dtype(2.31))
    tol = 50 * np.finfo(gtype).eps
    assert np.isclose(assemble_scalar(f * ufl.dx), f.value, atol=tol)


@pytest.mark.parametrize(
    "gtype, dtype",
    [
        [np.float64, np.float64],
        [np.float64, np.complex128],
        [np.float32, np.float32],
        [np.float32, np.complex64],
    ],
)
def test_assemble_scalar_coefficient(gtype, dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=gtype)
    V = functionspace(mesh, ("Lagrange", 3))
    u = Function(V, dtype=dtype)
    u.interpolate(lambda x: 3 * x[0] + 2 * x[1] ** 3)
    tol = 50 * np.finfo(gtype).eps
    assert np.isclose(assemble_scalar(u * ufl.dx), gtype(3 / 2 + 2 / 4), atol=tol)


def test_incompatible_coeff_function():
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=np.float64)
    V = functionspace(mesh, ("Lagrange", 3))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: 3 * x[0] + 2 * x[1] ** 3)
    c = Constant(mesh, np.complex128(2.31))
    with pytest.raises(
        RuntimeError, match="All coefficients and constants must have the same data type"
    ):
        assemble_scalar(u * c * ufl.dx)


@pytest.mark.parametrize(
    "gtype, dtype",
    [
        [np.float64, np.float64],
        [np.float64, np.complex128],
        [np.float32, np.float32],
        [np.float32, np.complex64],
    ],
)
@pytest.mark.parametrize("norm_type", ["L2", "H1", "H10", "l2"])
def test_norm(norm_type, dtype, gtype):
    if norm_type == "l2":
        pytest.xfail("Unexpected norm type")

    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=gtype)
    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V, dtype=dtype)
    u.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)

    x = ufl.SpatialCoordinate(mesh)
    expr = u - ufl.sin(ufl.pi * x[0])
    compiled_norm = norm(expr, norm_type)

    result = assemble_scalar(compiled_norm)

    if norm_type == "L2":
        ref_form = ufl.inner(expr, expr) * ufl.dx
    elif norm_type == "H1":
        ref_form = (
            ufl.inner(expr, expr) * ufl.dx + ufl.inner(ufl.grad(expr), ufl.grad(expr)) * ufl.dx
        )
    elif norm_type == "H10":
        ref_form = ufl.inner(ufl.grad(expr), ufl.grad(expr)) * ufl.dx

    reference = assemble_scalar(ref_form)
    tol = 50 * np.finfo(dtype).eps
    assert np.isclose(result, reference, atol=tol)
