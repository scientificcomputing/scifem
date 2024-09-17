from mpi4py import MPI
import dolfinx
import ufl
import numpy as np
import scifem


def test_real_functionspace_stiffness_matrix():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    V = scifem.create_real_functionspace(mesh)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx

    A = dolfinx.fem.assemble_matrix(dolfinx.fem.form(a), bcs=[])
    assert np.allclose(A.to_dense(), 1.0)
