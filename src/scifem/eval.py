from mpi4py import MPI
import dolfinx
import numpy as np
import numpy.typing as npt


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
    assert (
        len(points.shape) == 2
    ), f"Expected points to have shape (num_points, dim), got {points.shape}"
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
