from mpi4py import MPI
import pytest
import dolfinx
import basix
import numpy as np
import ufl
from scifem import PointSource


def test_midpoint():
    """Check that adding point sources at all midpoints is equivalent of assembling
    the form ``v/abs(det(J))dx`` with a midpoint quadrature rule"""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    tdim = mesh.topology.dim
    num_cells_local = mesh.topology.index_map(tdim).size_local
    midpoints = dolfinx.mesh.compute_midpoints(
        mesh, tdim, np.arange(num_cells_local, dtype=np.int32)
    )
    # Send all points from rank 0
    source_points = mesh.comm.gather(midpoints, root=0)
    if mesh.comm.rank == 0:
        source_points = np.vstack(source_points)
    else:
        source_points = np.empty((0, 3), dtype=midpoints.dtype)
    point_source = PointSource(V, source_points)
    np.testing.assert_allclose(point_source._cells, np.arange(num_cells_local, dtype=np.int32))
    np.testing.assert_allclose(point_source._points, midpoints)

    b = dolfinx.fem.Function(V)
    b.x.array[:] = 0
    point_source.apply_to_vector(b)
    b.x.scatter_reverse(dolfinx.la.InsertMode.add)
    b.x.scatter_forward()

    cell_geometry = basix.geometry(mesh.basix_cell())
    midpoint = (np.sum(cell_geometry, axis=0) / midpoints.shape[1]).reshape(
        -1, cell_geometry.shape[1]
    )

    b_ref = dolfinx.fem.Function(V)
    b_ref.x.array[:] = 0
    v = ufl.TestFunction(V)
    dx = ufl.Measure(
        "dx",
        domain=mesh,
        metadata={
            "quadrature_points": midpoint,
            "quadrature_weights": np.array([1.0]),
            "quadrature_rule": "custom",
        },
    )
    ref_L = dolfinx.fem.form(ufl.conj(v) / abs(ufl.JacobianDeterminant(mesh)) * dx)
    dolfinx.fem.assemble_vector(b_ref.x.array, ref_L)
    b_ref.x.scatter_reverse(dolfinx.la.InsertMode.add)
    b_ref.x.scatter_forward()

    np.testing.assert_allclose(b.x.array, b_ref.x.array)


def test_outside():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))
    if mesh.comm.rank == 0:
        point = np.array([[1.1, 1.8, 0]])
    else:
        point = np.empty((0, 3), dtype=mesh.geometry.x.dtype)

    if MPI.COMM_WORLD.rank == 0:
        with pytest.raises(ValueError, match="Point source is outside the mesh"):
            ps = PointSource(V, point)
    else:
        # Can only catch this on a single rank with pytest as it is
        # the only one that sends in a point
        ps = PointSource(V, point)

    # Check if perturbing the mesh gets the point within
    mesh.geometry.x[:, :] *= 2
    b = dolfinx.fem.Function(V)
    ps = PointSource(V, point)
    ps.apply_to_vector(b, recompute=False)
    b.x.scatter_reverse(dolfinx.la.InsertMode.add)
    b.x.scatter_forward()

    # Check sanity of values in b
    bbtree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(bbtree, point)
    cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point)
    # Only use evaluate for points on current processor
    # BUG: In DOLFINx 0.8.0, links(0) yields a too long array
    if cells.offsets[-1] > 0:
        cell = cells.links(0)[0]
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x = mesh.geometry.cmap.pull_back(point.reshape(-1, 3), mesh.geometry.x[geom_dofs])
        ref_values = V.ufl_element().tabulate(0, ref_x).flatten()
        b_nonzero = np.flatnonzero(b.x.array)
        dofs = V.dofmap.cell_dofs(cell)
        assert len(b_nonzero) == len(dofs)
        assert len(np.intersect1d(b_nonzero, dofs)) == len(dofs)
        b_contributions = b.x.array[dofs]
        np.testing.assert_allclose(ref_values, b_contributions)

    # Scale down geometry and check that re-computing raises an error
    mesh.geometry.x[:, :] /= 2
    ps.apply_to_vector(b, recompute=False)

    if MPI.COMM_WORLD.rank == 0:
        with pytest.raises(ValueError, match="Point source is outside the mesh"):
            ps.apply_to_vector(b, recompute=True)
    else:
        ps.apply_to_vector(b, recompute=True)
