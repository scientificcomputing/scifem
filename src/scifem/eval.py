from mpi4py import MPI
import dolfinx
import numpy as np
import numpy.typing as npt

__all__ = ["evaluate_function"]


def evaluate_function(
    u: dolfinx.fem.Function, points: npt.ArrayLike, broadcast=True
) -> npt.NDArray[np.float64]:
    """Evaluate a function at a set of points.

    Args:
        u: The function to evaluate.
        points: The points to evaluate the function at.
        broadcast: If True, the values will be broadcasted to all processes.

            Note:
                Uses a global MPI call to broadcast values, thus this has to
                be called on all active processes synchronously.

            Note:
                If the function is discontinuous, different processes may return
                different values for the same point.
                In this case, the value returned is the maximum value across all processes.

    Returns:
        The values of the function evaluated at the points.


    """
    mesh = u.function_space.mesh
    u.x.scatter_forward()
    comm = mesh.comm
    points = np.array(points, dtype=np.float64)
    assert len(points.shape) == 2, (
        f"Expected points to have shape (num_points, dim), got {points.shape}"
    )
    num_points = points.shape[0]
    extra_dim = 3 - mesh.geometry.dim

    # Append zeros to points if the mesh is not 3D
    if extra_dim > 0:
        points = np.hstack((points, np.zeros((points.shape[0], extra_dim))))

    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    # Find cells whose bounding-box collide with the the points
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    adj = dolfinx.geometry.compute_colliding_cells(mesh, potential_colliding_cells, points)
    indices = np.flatnonzero(adj.offsets[1:] - adj.offsets[:-1])
    cells = adj.array[adj.offsets[indices]]
    points_on_proc = points[indices]

    values = u.eval(points_on_proc, cells)
    if broadcast:
        bs = u.function_space.dofmap.index_map_bs
        # Create array to store values and fill with -inf
        # to ensure that all points are included in the allreduce
        # with op=MPI.MAX
        u_out = np.ones((num_points, bs), dtype=np.float64) * -np.inf
        # Fill in values for points on this process
        u_out[indices, :] = values
        # Now loop over all processes and find the maximum value
        for i in range(num_points):
            if bs > 1:
                # If block size is larger than 1, loop over blocks
                for j in range(bs):
                    u_out[i, j] = comm.allreduce(u_out[i, j], op=MPI.MAX)
            else:
                u_out[i] = comm.allreduce(u_out[i], op=MPI.MAX)

        return u_out
    else:
        return values


if dolfinx.has_petsc4py:
    from petsc4py import PETSc

    __all__.append("create_pointwise_observation_matrix")

    def create_pointwise_observation_matrix(
        Vh: dolfinx.fem.FunctionSpace, points: np.ndarray
    ) -> PETSc.Mat:
        """
        Constructs a sparse observation matrix :math:`B (m*bs \times N)`
        such that :math:`d = B u`

        Args:
            Vh: dolfinx.fem.FunctionSpace
            points: numpy array of shape (m, 3) or (m, 2)

        Returns:
            petsc4py.PETSc.Mat: The assembled observation matrix.

        Note:
            If :code:`Vh` is a scalar space (bs=1):
                - B has :math:`m` rows.
                - d[i] is the value of u at point i.

            If :code:`Vh` is a vector space (bs > 1):
                - B has :math:`m \times bs` rows.
                - :math:`d[i \times bs + c]` is the value of component :math:`c`
                    of :math:`u` at point :math:`i`.
        Note:
            Points outside the domain will lead to zero rows in the matrix.
        """
        mesh = Vh.mesh
        comm = mesh.comm

        # Make sure points are all in 3D for search
        points = np.ascontiguousarray(points, dtype=np.float64)
        original_dim = points.shape[1]
        if original_dim == 2:
            points_3d = np.zeros((points.shape[0], 3))
            points_3d[:, :2] = points
        else:
            points_3d = points

        # Find the cells containing the points
        bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points_3d)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, points_3d)

        # Setup PETSc Matrix
        # Number of observation points
        m_points = points.shape[0]
        # Number of DOFs in the FE space
        dofmap = Vh.dofmap
        index_map = dofmap.index_map
        bs = dofmap.bs
        # Number of rows and columns
        global_rows = m_points * bs
        n_dofs_global = index_map.size_global * bs
        local_dof_size = index_map.size_local * bs

        B = PETSc.Mat().create(comm)
        B.setSizes([[None, global_rows], [local_dof_size, n_dofs_global]])

        # Preallocation
        element = Vh.element

        # 5. Evaluation Data Structures
        geom_dofs = mesh.geometry.dofmap
        geom_x = mesh.geometry.x
        cmap = mesh.geometry.cmap

        # Iterate over all observation points and fill the matrix
        for i in range(m_points):
            # Get cells containing point i
            cells = colliding_cells.links(i)

            # If point not found on this process we continue
            # This means that points outside the domain will lead to zero rows
            if len(cells) == 0:
                continue

            # We take the first cell containing the point
            cell_index = cells[0]

            # Pull back point to reference coordinates
            cell_geom_indices = geom_dofs[cell_index]
            cell_coords = geom_x[cell_geom_indices]
            point_ref = cmap.pull_back(np.array([points_3d[i]]), cell_coords)
            # Evaluate basis functions at reference point
            basis_values = element.basix_element.tabulate(0, point_ref)[0, 0, :]

            # Get Global Indices
            local_dofs = dofmap.cell_dofs(cell_index)
            global_block_indices = index_map.local_to_global(local_dofs).astype(np.int32)

            # Insert values into matrix
            if bs == 1:
                # Scalar Case: Row i, Cols global_block_indices
                B.setValues(
                    [i],
                    global_block_indices,
                    basis_values,
                    addv=PETSc.InsertMode.INSERT_VALUES,
                )
            else:
                # Vector Case:
                # We have 'bs' components. We populate 'bs' rows for this single point.
                # Row (i*bs + 0) observes component 0 -> uses cols (global_block * bs + 0)
                # Row (i*bs + 1) observes component 1 -> uses cols (global_block * bs + 1)
                for comp in range(bs):
                    row_idx = i * bs + comp

                    # The columns for this component
                    # global_block_indices are the node indices.
                    # The actual matrix column is node_index * bs + comp
                    col_indices = global_block_indices * bs + comp

                    # The weights are simply the scalar basis function values
                    B.setValues(
                        [row_idx],
                        col_indices.astype(np.int32),
                        basis_values,
                        addv=PETSc.InsertMode.INSERT_VALUES,
                    )

        B.assemble()
        return B
