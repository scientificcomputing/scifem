import pytest
import builtins
import basix.ufl
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import dolfinx
from scifem import evaluate_function, compute_extrema
import ufl


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
def test_evaluate_scalar_function_2D(cell_type):
    comm = MPI.COMM_WORLD
    Lx = Ly = 2.0
    nx = ny = 10

    mesh = dolfinx.mesh.create_rectangle(
        comm=comm,
        points=[np.array([0.0, 0.0]), np.array([Lx, Ly])],
        n=[nx, ny],
        cell_type=cell_type,
    )

    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    u = dolfinx.fem.Function(V, dtype=PETSc.ScalarType)
    f = lambda x: x[0] + 2 * x[1]
    u.interpolate(f)

    points = np.array([[0.0, 0.0], [0.2, 0.2], [0.5, 0.5], [0.7, 0.2]])
    u_values = evaluate_function(u, points)
    exact = np.array(f(points.T)).T

    assert np.allclose(u_values[:, 0], exact)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
def test_evaluate_vector_function_2D(cell_type):
    comm = MPI.COMM_WORLD
    Lx = Ly = 2.0
    nx = ny = 10

    mesh = dolfinx.mesh.create_rectangle(
        comm=comm,
        points=[np.array([0.0, 0.0]), np.array([Lx, Ly])],
        n=[nx, ny],
        cell_type=cell_type,
    )

    V = dolfinx.fem.functionspace(mesh, ("P", 1, (2,)))
    u = dolfinx.fem.Function(V, dtype=PETSc.ScalarType)
    f = lambda x: (x[0], 2 * x[1])
    u.interpolate(f)

    points = np.array([[0.0, 0.0], [0.2, 0.2], [0.5, 0.5], [0.7, 0.2]])
    u_values = evaluate_function(u, points)
    exact = np.array(f(points.T)).T

    assert np.allclose(u_values, exact)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
def test_evaluate_scalar_function_3D(cell_type):
    comm = MPI.COMM_WORLD
    Lx = Ly = Lz = 2.0
    nx = ny = nz = 10

    mesh = dolfinx.mesh.create_box(
        comm=comm,
        points=[np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        n=[nx, ny, nz],
        cell_type=cell_type,
    )

    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    u = dolfinx.fem.Function(V, dtype=PETSc.ScalarType)
    f = lambda x: x[0] + 2 * x[1]
    u.interpolate(f)

    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.3], [0.5, 0.5, 0.2], [0.7, 0.2, 0.5]])
    u_values = evaluate_function(u, points)
    exact = np.array(f(points.T)).T

    assert np.allclose(u_values[:, 0], exact)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
def test_evaluate_vector_function_3D(cell_type):
    comm = MPI.COMM_WORLD
    Lx = Ly = Lz = 2.0
    nx = ny = nz = 10

    mesh = dolfinx.mesh.create_box(
        comm=comm,
        points=[np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        n=[nx, ny, nz],
        cell_type=cell_type,
    )

    V = dolfinx.fem.functionspace(mesh, ("P", 1, (3,)))
    u = dolfinx.fem.Function(V, dtype=PETSc.ScalarType)
    f = lambda x: (x[0], 2 * x[1], -x[2])
    u.interpolate(f)

    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.3], [0.5, 0.5, 0.2], [0.7, 0.2, 0.5]])
    u_values = evaluate_function(u, points)
    exact = np.array(f(points.T)).T

    assert np.allclose(u_values, exact)


@pytest.mark.parametrize("degree", [5])
@pytest.mark.parametrize("extrema", [min, max])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_extrema_func(cell_type, extrema, dtype, degree: int):
    tol = 10 * np.finfo(dtype).eps

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 7, 7, 7, cell_type=cell_type, dtype=dtype)
    el = basix.ufl.element("Lagrange", mesh.basix_cell(), degree, dtype=dtype)
    V = dolfinx.fem.functionspace(mesh, el)

    x_p = np.array([0.318, 0.852, 0.53], dtype=dtype)

    sign = 1 if extrema is builtins.min else -1

    def f(x):
        return (
            -0.8
            * sign
            * np.exp(-sum([(x[i] - x_p[i]) ** 2 for i in range(mesh.geometry.dim)]) / 0.1)
        )

    u = dolfinx.fem.Function(V, dtype=dtype)
    u.interpolate(f)

    u_ex, _X_ex = compute_extrema(u, extrema)

    assert np.isclose(u_ex, -sign * 0.8, atol=tol)

    x_ufl = ufl.SpatialCoordinate(mesh)
    x_p_ufl = ufl.as_vector(x_p)
    f_ufl = -0.8 * sign * ufl.exp(-ufl.dot(x_ufl - x_p_ufl, x_ufl - x_p_ufl) / 0.1)

    u_ufl_ex, _X_ufl_ex = compute_extrema(f_ufl, extrema)

    assert np.isclose(u_ufl_ex, -sign * 0.8, atol=tol)
    np.testing.assert_allclose(_X_ufl_ex, x_p, atol=tol)
