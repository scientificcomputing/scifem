from mpi4py import MPI
import dolfinx
import scifem.interpolation
import pytest
import ufl
import numpy as np

@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral, dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron])
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
    u.interpolate(lambda x: x[0]**degree + x[1] if tdim == 2 else x[0] + x[1] + x[2]**degree)


    q = dolfinx.fem.Function(Q)
    expr = ufl.TrialFunction(V)
    if use_petsc:
        A = scifem.interpolation.petsc_interpolation_matrix(expr, Q)
        A.mult(u.x.petsc_vec, q.x.petsc_vec)
    else:
        A = scifem.interpolation.interpolation_matrix(expr, Q)
        A.mult(u.x, q.x)
        q.x.scatter_forward()
        mesh.comm.barrier()
        print(A.data)
        print(u.x.array, q.x.array)
       
    q.x.scatter_forward()

    # Z = dolfinx.fem.interpolation_matrix(V, Q)
    # Z_petsc = dolfinx.fem.petsc.interpolation_matrix(V, Q)
    # Z_petsc.assemble()

    # q_z_ptsc = dolfinx.fem.Function(Q)
    # Z_petsc.mult(u.x.petsc_vec, q_z_ptsc.x.petsc_vec)

    # q_z = dolfinx.fem.Function(Q)
    # Z.mult(u.x, q_z.x)
    # q_z.x.scatter_forward()
    q_ref = dolfinx.fem.Function(Q)
    q_ref.interpolate(u)

    np.testing.assert_allclose(q.x.array, q_ref.x.array, rtol=1e-12, atol=1e-13)

    print(q.x.array)    