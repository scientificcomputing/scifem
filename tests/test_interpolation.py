from mpi4py import MPI
import dolfinx
import scifem.interpolation
import pytest
import ufl
import numpy as np


@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("degree", [1, 3, 5])
def test_interpolation_matrix(use_petsc, cell_type, degree):
    if use_petsc:
        pytest.importorskip("petsc4py")

    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type=cell_type)
    else:
        raise ValueError("Unsupported cell type")

    V = dolfinx.fem.functionspace(mesh, ("DG", degree))
    Q = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] ** degree + x[1] if tdim == 2 else x[0] + x[1] + x[2] ** degree)

    q = dolfinx.fem.Function(Q)
    expr = ufl.TrialFunction(V)
    if use_petsc:
        A = scifem.interpolation.petsc_interpolation_matrix(expr, Q)
        A.mult(u.x.petsc_vec, q.x.petsc_vec)
        A.destroy()
    else:
        A = scifem.interpolation.interpolation_matrix(expr, Q)
        # Built in matrices has to use a special input vector, with additional ghosts.
        _x = dolfinx.la.vector(A.index_map(1), A.block_size[1])
        num_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        _x.array[:num_owned_dofs] = u.x.array[:num_owned_dofs]
        _x.scatter_forward()
        if not hasattr(dolfinx.la.MatrixCSR, "mult"):
            pytest.skip("MatrixCSR has no mult method")
        A.mult(_x, q.x)

    q.x.scatter_forward()

    q_ref = dolfinx.fem.Function(Q)
    q_ref.interpolate(u)

    np.testing.assert_allclose(q.x.array, q_ref.x.array, rtol=1e-12, atol=1e-13)


@pytest.mark.skipif(
    not hasattr(dolfinx.fem, "discrete_gradient"),
    reason="Cannot verify without discrete gradient from DOLFINx",
)
@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("degree", [1, 3, 5])
def test_discrete_gradient(degree, use_petsc, cell_type):
    if use_petsc:
        pytest.importorskip("petsc4py")

    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type=cell_type)
    else:
        raise ValueError("Unsupported cell type")

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    W = dolfinx.fem.functionspace(mesh, ("Nedelec 1st kind H(curl)", degree))

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] ** degree + x[1])

    w = dolfinx.fem.Function(W)
    expr = ufl.grad(ufl.TrialFunction(V))

    G_ref = dolfinx.fem.discrete_gradient(V, W)

    # Built in matrices has to use a special input vector, with additional ghosts.
    try:
        _x = dolfinx.la.vector(G_ref.index_map(1), G_ref.block_size[1])
    except AttributeError:
        # Bug in DOLFINx 0.9.0
        _x = dolfinx.la.vector(G_ref.index_map(1), G_ref.bs[1])

    num_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    _x.array[:num_owned_dofs] = u.x.array[:num_owned_dofs]
    _x.scatter_forward()

    if use_petsc:
        A = scifem.interpolation.petsc_interpolation_matrix(expr, W)
        A.mult(u.x.petsc_vec, w.x.petsc_vec)
        A.destroy()
    else:
        if not hasattr(dolfinx.la.MatrixCSR, "mult"):
            pytest.skip("MatrixCSR has no mult method")
        A = scifem.interpolation.interpolation_matrix(expr, W)
        A.mult(_x, w.x)
    w.x.scatter_forward()

    w_ref = dolfinx.fem.Function(W)
    if not hasattr(dolfinx.la.MatrixCSR, "mult"):
        # Fallback to PETSc discrete gradient on 0.9
        pytest.mark.skipif(not dolfinx.has_petsc4py, reason="Cannot verify without petsc4py")
        import dolfinx.fem.petsc as _petsc

        G_ref = _petsc.discrete_gradient(V, W)
        G_ref.assemble()
        G_ref.mult(u.x.petsc_vec, w_ref.x.petsc_vec)
    else:
        G_ref.mult(_x, w_ref.x)
    w_ref.x.scatter_forward()

    np.testing.assert_allclose(w.x.array, w_ref.x.array, rtol=1e-11, atol=1e-12)


@pytest.mark.skipif(
    not hasattr(dolfinx.fem, "discrete_curl"),
    reason="Cannot verify without discrete curl from DOLFINx",
)
@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("degree", [1, 2])
def test_discrete_curl(degree, use_petsc, cell_type):
    if use_petsc:
        pytest.importorskip("petsc4py")

    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type=cell_type)
    else:
        raise ValueError("Unsupported cell type")

    V = dolfinx.fem.functionspace(mesh, ("Nedelec 2nd kind H(curl)", degree + 1))
    W = dolfinx.fem.functionspace(mesh, ("RT", degree))

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: (x[0] ** degree, x[1] ** degree - 1, -x[2]))

    w = dolfinx.fem.Function(W)
    expr = ufl.curl(ufl.TrialFunction(V))

    G_ref = dolfinx.fem.discrete_curl(V, W)

    # Built in matrices has to use a special input vector, with additional ghosts.
    _x = dolfinx.la.vector(G_ref.index_map(1), G_ref.block_size[1])
    num_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    _x.array[:num_owned_dofs] = u.x.array[:num_owned_dofs]
    _x.scatter_forward()

    if use_petsc:
        A = scifem.interpolation.petsc_interpolation_matrix(expr, W)
        A.mult(u.x.petsc_vec, w.x.petsc_vec)
        A.destroy()
    else:
        if not hasattr(dolfinx.la.MatrixCSR, "mult"):
            pytest.skip("MatrixCSR has no mult method")
        A = scifem.interpolation.interpolation_matrix(expr, W)
        A.mult(_x, w.x)
    w.x.scatter_forward()

    w_ref = dolfinx.fem.Function(W)
    G_ref.mult(_x, w_ref.x)
    w_ref.x.scatter_forward()

    np.testing.assert_allclose(w.x.array, w_ref.x.array, rtol=1e-10, atol=1e-11)
