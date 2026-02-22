import pytest

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import dolfinx

from scifem import evaluate_function, create_pointwise_observation_matrix


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


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
def test_pointwise_observation_2d_scalar(cell_type):
    """Test 2D scalar function with block size 1 on unit square."""
    # 1. Mesh & Space
    comm = MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_square(comm, 10, 10, cell_type=cell_type)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

    # 2. Function u = 2x + y
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 2 * x[0] + x[1])

    # 3. Observation Points
    points = np.array(
        [
            [0.5, 0.5, 0.0],  # Expected: 2(0.5) + 0.5 = 1.5
            [0.0, 0.0, 0.0],  # Expected: 0
            [1.0, 0.2, 0.0],  # Expected: 2(1) + 0.2 = 2.2
        ]
    )

    # 4. Create Matrix
    B = create_pointwise_observation_matrix(V, points)

    # 5. Compute d = B * u
    d = B.createVecLeft()
    B.mult(u.x.petsc_vec, d)

    local_vals = d.array
    ranges = B.getOwnershipRange()  # (start, end) row indices owned by this rank

    expected_full = np.array([1.5, 0.0, 2.2])

    for local_idx, global_idx in enumerate(range(ranges[0], ranges[1])):
        assert np.isclose(local_vals[local_idx], expected_full[global_idx], atol=1e-10)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
def test_pointwise_observation_3d_scalar(cell_type):
    comm = MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_cube(comm, 5, 5, 5, cell_type=cell_type)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))

    u = dolfinx.fem.Function(V)
    # u(x,y,z) = x + y + z
    u.interpolate(lambda x: x[0] + x[1] + x[2])

    points = np.array(
        [
            [0.5, 0.5, 0.5],  # Expected: 1.5
            [0.1, 0.1, 0.1],  # Expected: 0.3
        ]
    )

    B = create_pointwise_observation_matrix(V, points)
    d = B.createVecLeft()
    B.mult(u.x.petsc_vec, d)

    ranges = B.getOwnershipRange()
    local_vals = d.array
    expected_full = np.array([1.5, 0.3])

    for local_idx, global_idx in enumerate(range(ranges[0], ranges[1])):
        assert np.isclose(local_vals[local_idx], expected_full[global_idx], atol=1e-10)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
def test_pointwise_observation_2d_vector_bs2(cell_type):
    """Test 2D vector function with block size 2 on unit square."""
    comm = MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_square(comm, 5, 5, cell_type=cell_type)
    # Vector Function Space
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))

    u = dolfinx.fem.Function(V)
    # Set u = (x, y)
    u.interpolate(lambda x: (x[0], x[1]))

    points = np.array([[0.5, 0.8, 0.0], [0.2, 0.3, 0.0]])

    # B should be size (num_points * 2) x N
    B = create_pointwise_observation_matrix(V, points)
    d = B.createVecLeft()
    B.mult(u.x.petsc_vec, d)

    # Rows are interleaved:
    # Row 0: Pt0, Comp X -> 0.5
    # Row 1: Pt0, Comp Y -> 0.8
    # Row 2: Pt1, Comp X -> 0.2
    # Row 3: Pt1, Comp Y -> 0.3
    expected_full = np.array([0.5, 0.8, 0.2, 0.3])

    ranges = B.getOwnershipRange()
    local_vals = d.array

    for local_idx, global_idx in enumerate(range(ranges[0], ranges[1])):
        assert np.isclose(local_vals[local_idx], expected_full[global_idx], atol=1e-10)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
def test_pointwise_observation_3d_vector_bs3(cell_type):
    """Test 3D vector function with block size 3 on unit cube."""
    comm = MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_cube(comm, 4, 4, 4, cell_type=cell_type)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: (x[2], x[1], x[0]))

    points = np.array([[0.1, 0.5, 0.9]])

    # Expected:
    # x=0.1, y=0.5, z=0.9
    # u_x = z = 0.9
    # u_y = y = 0.5
    # u_z = x = 0.1
    expected_full = np.array([0.9, 0.5, 0.1])

    B = create_pointwise_observation_matrix(V, points)
    d = B.createVecLeft()
    B.mult(u.x.petsc_vec, d)

    ranges = B.getOwnershipRange()
    local_vals = d.array

    for local_idx, global_idx in enumerate(range(ranges[0], ranges[1])):
        assert np.isclose(local_vals[local_idx], expected_full[global_idx], atol=1e-10)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
def test_pointwise_observation_points_outside(cell_type):
    """Test behavior when points are outside the domain."""
    comm = MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_square(comm, 5, 5, cell_type=cell_type)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.ones_like(x[0]))  # u=1 everywhere

    points = np.array(
        [
            [2.0, 2.0, 0.0],  # Outside
            [0.5, 0.5, 0.0],  # Inside
        ]
    )

    B = create_pointwise_observation_matrix(V, points)
    d = B.createVecLeft()
    B.mult(u.x.petsc_vec, d)

    # Assume points outside the domain yield zero values
    expected_full = np.array([0.0, 1.0])

    ranges = B.getOwnershipRange()
    local_vals = d.array

    for local_idx, global_idx in enumerate(range(ranges[0], ranges[1])):
        assert np.isclose(local_vals[local_idx], expected_full[global_idx], atol=1e-10)
