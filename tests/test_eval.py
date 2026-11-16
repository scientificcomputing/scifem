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


@pytest.mark.parametrize("shape", [(), (2,)])
def test_evaluate_function_keeps_nan(shape):
    # NaN must come back as NaN on any number of ranks, wherever the point is owned
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("P", 1, shape))
    u = dolfinx.fem.Function(V)
    u.x.array[:] = np.nan

    x = np.linspace(0.05, 0.95, 5)
    points = np.array([[xi, yi] for xi in x for yi in x])
    u_values = evaluate_function(u, points)

    assert u_values.shape == (len(points), int(np.prod(shape)))
    assert np.isnan(u_values).all()


def test_evaluate_function_value_size_differs_from_block_size():
    # Raviart-Thomas: block size 1, value size 2
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("RT", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: (np.ones_like(x[0]), 2 * np.ones_like(x[0])))

    u_values = evaluate_function(u, np.array([[0.3, 0.4], [0.8, 0.1]]))

    assert np.allclose(u_values, [[1.0, 2.0], [1.0, 2.0]])


def test_evaluate_function_nan_next_to_values():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.where(x[0] < 0.49, np.nan, x[0] + 2 * x[1]))

    x = np.linspace(0.05, 0.95, 7)
    points = np.array([[xi, yi] for xi in x for yi in x])
    u_values = evaluate_function(u, points)[:, 0]

    left = points[:, 0] < 0.37  # every vertex of the containing cell is NaN
    right = points[:, 0] > 0.63  # no vertex of the containing cell is NaN
    assert np.isnan(u_values[left]).all()
    assert np.allclose(u_values[right], points[right, 0] + 2 * points[right, 1])


def test_evaluate_function_point_outside_mesh_is_nan():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("P", 1))
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 1.0

    u_values = evaluate_function(u, np.array([[0.5, 0.5], [2.0, 2.0]]))[:, 0]

    assert np.isclose(u_values[0], 1.0)
    assert np.isnan(u_values[1])


def test_evaluate_function_same_result_on_all_ranks_for_discontinuous_function():
    # Points on facets lie in cells of several ranks; every rank must return the same value
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.sin(10 * x[0]) + x[1])

    x = np.linspace(0.0, 1.0, 9)  # the vertices of the mesh
    points = np.array([[xi, yi] for xi in x for yi in x])
    u_values = evaluate_function(u, points)

    all_values = comm.allgather(u_values)
    for other in all_values:
        np.testing.assert_array_equal(other, all_values[0])
    assert np.isfinite(u_values).all()


@pytest.mark.parametrize("degree", [5])
@pytest.mark.parametrize("extrema", [min, max])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_extrema_func(cell_type, extrema, dtype, degree: int):
    tol = 100 * np.finfo(dtype).eps

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6, cell_type=cell_type, dtype=dtype)
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

    u_ex, _X_ex = compute_extrema(u, extrema, tol=tol)

    assert np.isclose(u_ex, -sign * 0.8, atol=tol)

    x_ufl = ufl.SpatialCoordinate(mesh)
    x_p_ufl = ufl.as_vector(x_p)
    f_ufl = -0.8 * sign * ufl.exp(-ufl.dot(x_ufl - x_p_ufl, x_ufl - x_p_ufl) / 0.1)

    u_ufl_ex, _X_ufl_ex = compute_extrema(f_ufl, extrema, tol=tol)

    assert np.isclose(u_ufl_ex, -sign * 0.8, atol=tol)
    np.testing.assert_allclose(_X_ufl_ex, x_p, atol=tol)


def test_extrema_requires_kind():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    x = ufl.SpatialCoordinate(mesh)
    with pytest.raises(TypeError, match="kind"):
        compute_extrema(x[0])


def test_extrema_unknown_kind():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    x = ufl.SpatialCoordinate(mesh)
    with pytest.raises(ValueError, match=f"Unknown extrema {sum}"):
        compute_extrema(x[0], sum)
