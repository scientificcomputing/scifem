import pytest
import builtins
import basix.ufl
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import dolfinx
from scifem import evaluate_function, compute_extrema


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


@pytest.mark.parametrize("degree", [6])
@pytest.mark.parametrize("extrema", [min, max])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_extrema_func(cell_type, extrema, dtype, degree: int):
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, cell_type=cell_type, dtype=dtype)
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

    u_ex, X_ex = compute_extrema(u, extrema, options={"disp": True})

    with dolfinx.io.VTXWriter(mesh.comm, "u.bp", [u]) as bp:
        bp.write(0.0)

    tol = 10 * np.finfo(dtype).eps
    # np.testing.assert_allclose(X_ex, x_p,atol=tol)
    assert np.isclose(u_ex, -sign * 0.8, atol=tol)

    # breakpoint()
