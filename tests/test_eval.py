import pytest

from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import dolfinx

from scifem import evaluate_function


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
