from mpi4py import MPI
import pytest
import dolfinx
from scifem import vertex_to_dofmap, dof_to_vertexmap
import numpy as np


@pytest.mark.parametrize("degree", range(1, 4))
@pytest.mark.parametrize("value_size", [(), (2,), (2, 3)])
def test_vertex_to_dofmap_P(degree, value_size):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 2)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, value_size))

    v_to_d = vertex_to_dofmap(V)
    mesh.topology.create_connectivity(0, mesh.topology.dim)
    geom_indices = dolfinx.mesh.entities_to_geometry(
        mesh, 0, np.arange(len(v_to_d), dtype=np.int32)
    )

    x_V = V.tabulate_dof_coordinates()
    x_g = mesh.geometry.x
    tol = 1e3 * np.finfo(x_g.dtype).eps
    mesh = V.mesh

    np.testing.assert_allclose(x_V[v_to_d], x_g[geom_indices.flatten()], atol=tol)

    # Test inverse map
    d_to_v = dof_to_vertexmap(V)
    np.testing.assert_allclose(d_to_v[v_to_d], np.arange(len(v_to_d)))

    # Mark all dofs connected with a vertex
    dof_marker = np.full(len(d_to_v), True, dtype=np.bool_)
    dof_marker[v_to_d] = False

    other_dofs = np.flatnonzero(dof_marker)
    np.testing.assert_allclose(d_to_v[other_dofs], -1)
