from mpi4py import MPI
import numpy as np
import numpy.typing as npt
import dolfinx

mpi_dtype = {
    np.float64: MPI.DOUBLE,
    np.float32: MPI.FLOAT,
    np.int32: MPI.INT32_T,
    np.int64: MPI.INT64_T,
    np.complex128: MPI.DOUBLE_COMPLEX,
    np.complex64: MPI.COMPLEX,
}


def all_to_allv(comm, send_data, num_send_data, recv_data, num_recv_data):
    dtype = mpi_dtype[send_data.dtype.type]
    assert recv_data.dtype == send_data.dtype, (
        f"Data types do not match, {recv_data.dtype} != {send_data.dtype}"
    )
    assert (d_size := send_data.size) == (s_size := num_send_data.sum()), (
        f"Number of send data {d_size}  does not match data size {s_size}"
    )
    assert (d_size := recv_data.size) == (r_size := num_recv_data.sum()), (
        f"Number of recv data {d_size}  does not match data size {r_size}"
    )

    send_msg = [send_data, num_send_data, dtype]
    recv_msg = [recv_data, num_recv_data, dtype]
    comm.Neighbor_alltoallv(send_msg, recv_msg)


def all_to_all(comm, send_data, recv_data):
    """
    Exchange a single item with each neighbor in a distributed graph communicator.

    Note:
        The count is passed explicitly, and is 1 on every process. MPI-4.1 9.6.2 requires
        the type signature of ``sendcount``/``sendtype`` at a process to equal that of
        ``recvcount``/``recvtype`` at *any other* process in the communicator, not just at
        its neighbors, so the count must be identical on every process whatever its degree.
        Left implicit, mpi4py derives it as ``buffer size // degree`` and falls back to the
        whole buffer when the degree is zero, making it rank-local: 1 where the degree is
        nonzero, 0 where it is zero. Such a call is erroneous; Open MPI rejects it with
        ``MPI_ERR_TRUNCATE`` while MPICH happens to accept it. See
        https://github.com/open-mpi/ompi/issues/14452 for the discussion.

        ``all_to_allv`` is not affected: the vector variant is only required to match
        pairwise along each edge, so per-process counts may legitimately differ there.
    """
    dtype = mpi_dtype[send_data.dtype.type]
    assert recv_data.dtype == send_data.dtype, (
        f"Data types do not match, {recv_data.dtype} != {send_data.dtype}"
    )
    indegree, outdegree, _ = comm.Get_dist_neighbors_count()
    assert (d_size := send_data.size) == outdegree, (
        f"Number of send data {d_size} does not match number of destinations {outdegree}"
    )
    assert (d_size := recv_data.size) == indegree, (
        f"Number of recv data {d_size} does not match number of sources {indegree}"
    )
    comm.Neighbor_alltoall([send_data, 1, dtype], [recv_data, 1, dtype])


def get_ownership(imap) -> npt.NDArray[np.int32]:
    """Get ownership of each index in an index map."""
    owners = np.full(imap.size_local + imap.num_ghosts, imap.comm.rank, dtype=np.int32)
    owners[imap.size_local :] = imap.owners
    return owners


def find_position(data, values):
    """
    Find the position in values of each entry in data

    Example:

        .. highlight:: python
        .. code-block:: python

            values = np.array([4, 5, 1, 3, 2], dtype=np.int32)
            data = np.array([1, 2, 3, 4, 5, 2, 1], dtype=np.int32)
            b = find_position(data, values) # [2,4,3,0,1,4 2]

    Note:
        Where ``values`` repeats an entry, the first occurrence is returned. Uses a sorted
        search rather than a dense ``len(data) x len(values)`` comparison, so the cost is
        ``O((n + m) log m)`` in time and ``O(n + m)`` in memory.
    """
    if len(data) == 0:
        return np.zeros(0, dtype=np.int32)
    # a stable sort makes `searchsorted` land on the first of any repeated value
    order = np.argsort(values, kind="stable")
    slot = np.searchsorted(values, data, sorter=order)
    if np.any(slot >= len(values)):
        raise ValueError("find_position: data contains values not present in values")
    position = order[slot]
    if not np.array_equal(values[position], data):
        raise ValueError("find_position: data contains values not present in values")
    return position.astype(np.int32)


def compute_insert_position(
    data_owner: npt.NDArray[np.int32],
    destination_ranks: npt.NDArray[np.int32],
    out_size: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """
    Giving a list of ranks, compute the local insert position for each rank in a list
    sorted by destination ranks. This function is used for packing data from a
    given process to its destination processes.

    Example:

        .. highlight:: python
        .. code-block:: python

            data_owner = [0, 1, 1, 0, 2, 3]
            destination_ranks = [2,0,3,1]
            out_size = [1, 2, 1, 2]
            insert_position = compute_insert_position(data_owner, destination_ranks, out_size)

        Insert position is then ``[1, 4, 5, 2, 0, 3]``

    Note:
        Uses a sorted search rather than a dense ``len(data_owner) x
        len(destination_ranks)`` comparison, so the cost is ``O(n log n)`` in time and
        ``O(n)`` in memory.
    """
    if len(data_owner) == 0:
        return np.zeros(0, dtype=np.int32)
    # which destination block each item belongs to
    block = find_position(data_owner, destination_ranks)

    # Compute offsets for insertion based on input size
    send_offsets = np.zeros(len(out_size) + 1, dtype=np.intc)
    send_offsets[1:] = np.cumsum(out_size)
    assert send_offsets[-1] == len(data_owner)

    # Index of each item within its own block, in order of appearance. A stable sort by
    # block puts each block's items in a contiguous run, in their original order, so the
    # position within the run is the position within the block.
    order = np.argsort(block, kind="stable")
    within_block = np.empty(len(block), dtype=np.int64)
    within_block[order] = np.arange(len(block)) - np.repeat(send_offsets[:-1], out_size)

    return (within_block + send_offsets[block]).astype(np.int32)


def unroll_insert_position(
    insert_position: npt.NDArray[np.int32], block_size: int
) -> npt.NDArray[np.int32]:
    """
    Unroll insert position by a block size

    Example:


        .. highlight:: python
        .. code-block:: python

            insert_position = [1, 4, 5, 2, 0, 3]
            unrolled_ip = unroll_insert_position(insert_position, 3)

        where ``unrolled_ip = [3, 4 ,5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 0, 1, 2, 9, 10, 11]``
    """
    unrolled_ip = np.repeat(insert_position, block_size) * block_size
    unrolled_ip += np.tile(np.arange(block_size), len(insert_position))
    return unrolled_ip


def broadcast_marked_entities(mesh, dim, entities):
    """Extend a set of entities to every local copy of the entities in it.

    An entity marked on one process that holds it comes back marked on all of them. Use
    this on a set that is only correct on the owners, such as the output of
    `locate_entities_boundary` or `exterior_facet_indices`, when the ghost copies have to
    carry the same mark.

    Collective on the communicator of the index map for `dim`.

    Args:
        mesh: The mesh the entities belong to.
        dim: Topological dimension of `entities`.
        entities: Local indices of the marked entities, owned or ghost.

    Returns:
        Local indices of every entity marked on this process or on the owner of one of
        its entities, ascending.
    """
    marker = dolfinx.la.vector(mesh.topology.index_map(dim), 1, dtype=np.int32)
    marker.array[:] = 0
    marker.array[entities] = 1
    marker.scatter_reverse(dolfinx.la.InsertMode.add)
    marker.scatter_forward()
    return np.flatnonzero(marker.array).astype(np.int32)


def local_range(comm, num_indices):
    """The block of ``range(num_indices)`` this process is the post office for.

    Uses ``dolfinx.common.local_range`` where it exists, so that the blocks match what the
    rest of DOLFINx means by the same words; the fallback reproduces it.
    """
    if hasattr(dolfinx.common, "local_range"):
        return tuple(dolfinx.common.local_range(comm.rank, int(num_indices), comm.size))
    per_rank, remainder = divmod(int(num_indices), comm.size)
    low = comm.rank * per_rank + min(comm.rank, remainder)
    return low, low + per_rank + (1 if comm.rank < remainder else 0)


def index_owner(comm, indices, num_indices):
    """Which process is the post office for each of `indices`.

    The inverse of :func:`_local_range`, vectorised: the first ``num_indices % size``
    ranks hold one extra, so the blocks differ in length by at most one and no rank is
    left out.

    Args:
        comm: The communicator the blocks are spread over.
        indices: Indices in ``range(num_indices)``.
        num_indices: The size of the range.

    Returns:
        The rank responsible for each entry of `indices`.
    """
    per_rank, remainder = divmod(int(num_indices), comm.size)
    indices = np.asarray(indices, dtype=np.int64)
    split = remainder * (per_rank + 1)
    below = indices < split
    owner = np.empty(len(indices), dtype=np.int32)
    owner[below] = indices[below] // max(per_rank + 1, 1)
    owner[~below] = remainder + (indices[~below] - split) // max(per_rank, 1)
    return owner


def exchange_to_destinations(comm, dest_ranks, payload):
    """Send rows to the ranks that name them, and receive whatever arrives.

    Only the outgoing edges are known -- a process cannot tell in advance who will write
    to it -- so the neighbourhood is built with ``Create_dist_graph``, which derives the
    incoming edges from the outgoing ones, rather than ``Create_dist_graph_adjacent``.

    Collective.

    Args:
        comm: The communicator to exchange over.
        dest_ranks: Destination rank of each row of `payload`. Need not be sorted.
        payload: ``(n, k)`` of ``int64``, one row per entry of `dest_ranks`.

    Returns:
        ``(sources, received)``: the rank each received row came from, ascending, and the
        rows in that order.
    """
    # Not reshaped from a flat array: ``reshape(0, -1)`` is ambiguous, and a process with
    # nothing to send is the normal case here, not an edge case.
    payload = np.asarray(payload, dtype=np.int64)
    assert payload.ndim == 2 and len(payload) == len(dest_ranks)
    width = payload.shape[1]
    order = np.argsort(dest_ranks, kind="stable")
    dests, counts = np.unique(dest_ranks, return_counts=True)
    send_buffer = np.ascontiguousarray(payload[order])

    graph = comm.Create_dist_graph(
        [comm.rank], [len(dests)], dests.astype(np.int32).tolist(), MPI.UNWEIGHTED
    )
    try:
        in_ranks, _, _ = graph.Get_dist_neighbors()
        in_ranks = np.asarray(in_ranks, dtype=np.int32)

        # Uniform neighbourhood collective: the count is given explicitly and is the same
        # on every process, as MPI-4.1 9.6.2 requires. Letting mpi4py infer it from the
        # buffer size would make it rank-local, which is an erroneous call.
        recv_counts = np.zeros(len(in_ranks), dtype=np.int32)
        graph.Neighbor_alltoall(
            [counts.astype(np.int32), 1, MPI.INT32_T], [recv_counts, 1, MPI.INT32_T]
        )

        received = np.zeros((int(recv_counts.sum()), width), dtype=np.int64)
        graph.Neighbor_alltoallv(
            [send_buffer, counts.astype(np.int32) * width, MPI.INT64_T],
            [received, recv_counts * width, MPI.INT64_T],
        )
    finally:
        graph.Free()

    # `Get_dist_neighbors` lists the neighbours in the order MPI chose, not ascending, and
    # the receive buffer follows that order. Sort so the caller can rely on the grouping.
    sources = np.repeat(in_ranks, recv_counts).astype(np.int32)
    order = np.argsort(sources, kind="stable")
    return sources[order], received[order]
