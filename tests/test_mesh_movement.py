import pytest
import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
from dolfinx.fem import Function, functionspace
from scifem.mesh import move


@pytest.fixture
def mesh_fixture():
    """
    Creates a fresh unit square mesh for each test.
    """
    comm = MPI.COMM_WORLD
    # Create a simple 2D unit square mesh
    mesh = dolfinx.mesh.create_unit_square(comm, 5, 5)
    return mesh


def test_move_with_dolfinx_function(mesh_fixture):
    """
    Test moving the mesh with a dolfinx.fem.Function.
    """
    mesh = mesh_fixture
    gdim = mesh.geometry.dim

    # 1. Create a displacement function in a Vector FunctionSpace
    V = functionspace(mesh, ("Lagrange", 2, (gdim,)))
    displacement = Function(V)

    # Define a simple constant displacement: u = (0.1, 0.2)
    disp_val = np.array([0.1, 0.2])

    # Using interpolate to set constant value for test clarity
    displacement.interpolate(lambda x: np.full((gdim, x.shape[1]), disp_val[:, None]))

    # 2. Store original coords to compare later
    original_coords = mesh.geometry.x.copy()

    # 3. Apply move
    move(mesh, displacement)

    # 4. Verify
    tol = 10 * np.finfo(mesh.geometry.x.dtype).eps

    expected_coords = original_coords[:, : mesh.geometry.dim] + disp_val[None, : mesh.geometry.dim]
    np.testing.assert_allclose(
        mesh.geometry.x[:, : mesh.geometry.dim],
        expected_coords,
        rtol=tol,
        atol=tol,
        err_msg="Mesh did not move correctly using dolfinx.fem.Function",
    )


def test_move_with_ufl_expression(mesh_fixture):
    """
    Test moving the mesh with a UFL Expression.
    """
    mesh = mesh_fixture

    # 1. Define UFL expression
    x = ufl.SpatialCoordinate(mesh)
    # Shift nodes radially: u = 0.1 * x
    ufl_expr = 0.1 * x

    # 2. Store original coords
    original_coords = mesh.geometry.x.copy()

    # 3. Apply move
    move(mesh, ufl_expr)

    # 4. Verify
    # The expected new position is x_new = x_old + 0.1 * x_old = 1.1 * x_old
    expected_coords = original_coords * 1.1
    tol = 10 * np.finfo(mesh.geometry.x.dtype).eps
    np.testing.assert_allclose(
        mesh.geometry.x,
        expected_coords,
        rtol=tol,
        atol=tol,
        err_msg="Mesh did not move correctly using UFL Expression",
    )


def test_move_with_python_callable(mesh_fixture):
    """
    Test moving the mesh with a generic Python callable.
    """
    mesh = mesh_fixture

    # 1. Define python callable
    # Input x has shape (3, N) or (gdim, N)
    # Output must match shape
    def shear_displacement(x):
        # Shear mapping: dx = 0.5 * y, dy = 0
        vals = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=mesh.geometry.x.dtype)
        vals[0] = 0.5 * x[1]  # Displacement in x depends on y
        vals[1] = 0.0  # No displacement in y
        return vals

    # 2. Store original coords
    original_coords = mesh.geometry.x.copy()

    # 3. Apply move
    move(mesh, shear_displacement)

    # 4. Verify
    # Manually calculate expected shear
    expected_coords = original_coords.copy()
    expected_coords[:, 0] += 0.5 * original_coords[:, 1]

    tol = 10 * np.finfo(mesh.geometry.x.dtype).eps
    np.testing.assert_allclose(
        mesh.geometry.x,
        expected_coords,
        rtol=tol,
        atol=tol,
        err_msg="Mesh did not move correctly using Python callable",
    )


def test_move_accumulates_changes(mesh_fixture):
    """
    Ensure that calling move twice accumulates the displacement.
    """
    mesh = mesh_fixture
    # Move all coords by 0.1
    u = lambda x: np.full((mesh.geometry.dim, x.shape[1]), 0.1, dtype=mesh.geometry.x.dtype)
    original = mesh.geometry.x.copy()

    move(mesh, u)
    move(mesh, u)

    expected = original
    expected[:, : mesh.geometry.dim] += 0.2
    tol = 10 * np.finfo(mesh.geometry.x.dtype).eps
    np.testing.assert_allclose(
        mesh.geometry.x,
        expected,
        rtol=tol,
        atol=tol,
        err_msg="Multiple calls to move did not accumulate correctly",
    )
