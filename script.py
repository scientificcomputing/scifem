# Create a periodic mesh in parallel
# SPDX-License-Identifier: MIT
# Author: Jørgen S. Dokken

from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import numpy.typing as npt
import dataclasses
import inspect
import gmsh

mpi_dtype = {
    np.float64: MPI.DOUBLE,
    np.float32: MPI.FLOAT,
    np.int32: MPI.INT32_T,
    np.int64: MPI.INT64_T,
    np.complex128: MPI.DOUBLE_COMPLEX,
    np.complex64: MPI.COMPLEX,
}


def transfer_meshtags_to_periodic_mesh(
    mesh: dolfinx.mesh.Mesh,
    periodic_mesh: dolfinx.mesh.Mesh,
    replaced_vertices: npt.NDArray[np.int32],
    meshtags: dolfinx.mesh.MeshTags,
) -> dolfinx.mesh.MeshTags:
    """
    Transfer a mesh tag from a mesh to the periodic mesh.

    Note:
        Entities that have been replaced (vertices, edges, faces) are removed from the mesh tag

    Args:
        mesh: The original mesh
        periodic_mesh: The periodic mesh
        replaced_vertices: The vertices that have been replaced (local to process)
        meshtags: The mesh tag to transfer
    """

    # Remove entities that are fully replaced (all incident vertices replaced).
    if meshtags.dim != mesh.topology.dim:
        mesh.topology.create_connectivity(meshtags.dim, 0)
        e_to_v = mesh.topology.connectivity(meshtags.dim, 0)
        e_to_v_new = e_to_v.array.copy()
        replacement_indicator = np.isin(e_to_v_new, replaced_vertices)
        e_to_v_new[replacement_indicator] = -1
        new_adj = dolfinx.graph.adjacencylist(e_to_v_new, e_to_v.offsets)
        indices = []
        values = []
        for entity, value in zip(meshtags.indices, meshtags.values):
            # Keep entities with at least one retained vertex.
            if np.any(new_adj.links(entity) != -1):
                indices.append(entity)
                values.append(value)
        indices = np.array(indices, dtype=np.int32)
        values = np.array(values, dtype=meshtags.values.dtype)
    else:
        indices = meshtags.indices
        values = meshtags.values
    geom_indices = dolfinx.mesh.entities_to_geometry(mesh, meshtags.dim, indices)
    igi_indices = mesh.geometry.input_global_indices[geom_indices]

    periodic_mesh.topology.create_connectivity(
        mesh.topology.dim, 0
    )  # This should exist by default
    periodic_mesh.topology.create_entities(meshtags.dim)  # This has to be created
    periodic_mesh.topology.create_connectivity(
        meshtags.dim, 0
    )  # This is requried before distribute entity data
    local_entities, local_values = dolfinx.io.distribute_entity_data(
        periodic_mesh, meshtags.dim, igi_indices, values
    )
    adj = dolfinx.graph.adjacencylist(local_entities)
    return dolfinx.mesh.meshtags_from_entities(
        periodic_mesh, meshtags.dim, adj, local_values.astype(np.int32, copy=False)
    )


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
    """
    Get ownership of each index in an index map
    """
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


def _compat_index_map(comm, size_local, ghosts, owners, tag: int | None = None):
    if dolfinx.common.IndexMap != dolfinx.cpp.common.IndexMap:
        assert tag is not None, "Tag must be provided for dolfinx.common.index_map"
        return dolfinx.common.index_map(
            comm, size_local, ghosts=(ghosts, owners), tag=tag
        )
    else:
        try:
            return dolfinx.common.IndexMap(comm, size_local, ghosts, owners)
        except TypeError:
            assert tag is not None, "Tag must be provided for dolfinx.common.IndexMap"
            return dolfinx.common.IndexMap(comm, size_local, ghosts, owners, tag=tag)


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


def _extract_cpp_object(obj):
    if hasattr(obj, "_cpp_object"):
        return obj._cpp_object
    else:
        return obj


@dataclasses.dataclass
class VertexCorrespondence:
    """Which vertices of ``mesh`` are identified with which, and which ranks hold each end.

    This is the input of {py:func}`_build_periodic_mesh`, which consumes nothing else and
    never evaluates a coordinate.

    Stores the data of {py:class}`dolfinx.geometry.PointOwnershipData` for the
    `partner_vertex`, over query points that are the images of `indicator_vertices` -- the
    vertices given up to the partner side -- plus one extra array, `indicator_facets`, the
    facets given up with them.

    The two halves are keyed independently, so `indicator_vertices[i]` is *not* the vertex
    replaced by `partner_vertex[i]`: `indicator_vertices` and `src_owner` are keyed on what
    this process gives up, `dest_owner` and `partner_vertex` on what other processes gave up
    to it, and the two sides of a pair rarely live on the same process. Splitting a 6x6 unit
    square over three ranks gives one rank 7 and 0, and another 0 and 8. In serial the two
    lengths coincide, which makes the assumption easy to form and wrong to act on. They line
    up only after the exchange, which is what `compute_insert_position` reorders.

    Args:
        indicator_vertices: Local vertices, owned and ghost, that are to be replaced by
            their partner vertex. Broadened across processes: a vertex marked on its
            owner is marked on every process that ghosts it.
        indicator_facets: Local facets, owned and ghost, lying on the seam, i.e. the
            exterior facets all of whose vertices are in `indicator_vertices`. Broadened
            the same way.
        src_owner: For each entry of `indicator_vertices`, the rank owning *one* of the
            cells its partner vertex belongs to. A vertex is shared by several cells, so
            which one this names is arbitrary -- `_build_periodic_mesh` recovers the rest
            of the ranks holding a cell at that vertex, which all need the seam cells too.
            Note also that this is a *cell* owner, not the owner of the partner vertex,
            which the rank in question may merely ghost.
        dest_owner: For each vertex this process is the far side of, the rank that asked.
            Must be sorted ascending: the packing groups by destination and relies on it.
        partner_vertex: For each entry of `dest_owner`, the local vertex that replaces the
            vertex that rank gave up. Same length as `dest_owner`.
    """

    indicator_vertices: npt.NDArray[np.int32]
    indicator_facets: npt.NDArray[np.int32]
    src_owner: npt.NDArray[np.int32]
    dest_owner: npt.NDArray[np.int32]
    partner_vertex: npt.NDArray[np.int32]


def _match_vertices_geometric(
    mesh, indicator, mapping_function, max_chain_length: int | None = None
) -> VertexCorrespondence:
    """Pair up the vertices of the seam by evaluating `mapping_function` on them.

    Selects the seam with `indicator`, moves each selected vertex with `mapping_function`,
    and snaps the image onto the nearest vertex of the mesh, checking that it actually
    landed there.

    Args:
        mesh: The mesh to make periodic.
        indicator: Marks the vertices to be replaced, given coordinates as ``(3, n)``.
        mapping_function: Maps a marked vertex to the one it is identified with, given
            coordinates as ``(3, n)``.
        max_chain_length: How many times `mapping_function` may be re-applied to reach a
            vertex outside `indicator`, for a mapping that applies one offset per call and
            so needs several passes to carry a corner to its root. Defaults to
            ``mesh.topology.dim``, the number of directions such a mesh can be periodic
            in. Exceeding it raises, which is how a cyclic mapping is caught.

    Returns:
        The correspondence {py:func}`_build_periodic_mesh` consumes.
    """
    comm = mesh.comm
    if max_chain_length is None:
        max_chain_length = mesh.topology.dim

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # Find the vertices of the seam through the indicator function, and the facets given
    # up with them. Both are broadened so that the two sides agree on what is replaced.
    indicator_vertices = broadcast_marked_entities(
        mesh, 0, dolfinx.mesh.locate_entities_boundary(mesh, 0, indicator)
    )
    indicator_facets = broadcast_marked_entities(
        mesh,
        mesh.topology.dim - 1,
        dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, indicator),
    )

    geom_index = dolfinx.mesh.entities_to_geometry(mesh, 0, indicator_vertices).reshape(
        -1
    )
    owned_vertex_coords = mesh.geometry.x[geom_index]

    # A geometric tolerance has to be a length. `np.finfo(...).eps` describes relative
    # precision near 1.0, so as an absolute padding it stops meaning anything once the mesh
    # sits away from the origin: at coordinates around 1e6 a single representable double is
    # already 1.2e-10, larger than 10000 * eps. Take a small fraction of the smallest cell
    # instead, floored by the rounding error of the coordinates themselves, which grows with
    # distance from the origin.
    _cell_map = mesh.topology.index_map(mesh.topology.dim)
    _cell_sizes = dolfinx.cpp.mesh.h(
        mesh._cpp_object,
        mesh.topology.dim,
        np.arange(_cell_map.size_local + _cell_map.num_ghosts, dtype=np.int32),
    )
    h_min = comm.allreduce(
        _cell_sizes.min() if _cell_sizes.size else np.inf, op=MPI.MIN
    )
    coord_scale = comm.allreduce(
        np.abs(mesh.geometry.x).max() if mesh.geometry.x.size else 0.0, op=MPI.MAX
    )
    eps = max(
        1e-6 * h_min,
        100 * np.finfo(mesh.geometry.x.dtype).eps * max(1.0, coord_scale),
    )

    # Map vertices to new coordinates
    mapped_vertex_coords = np.ascontiguousarray(
        mapping_function(owned_vertex_coords.T).T
    )

    # Follow the mapping to a vertex outside `indicator`; see `max_chain_length` above.
    # Entirely local: `indicator` and `mapping_function` are pointwise in the coordinates,
    # so a process may run out of chains to follow before another does.
    local_unresolved = False
    for _ in range(max_chain_length):
        still_indicated = np.asarray(indicator(mapped_vertex_coords.T), dtype=np.bool_)
        if not still_indicated.any():
            break
        mapped_vertex_coords[still_indicated] = mapping_function(
            mapped_vertex_coords[still_indicated].T
        ).T
    else:
        local_unresolved = True

    # Collective check to check that all vertex mappings have been resolved.
    if comm.allreduce(int(local_unresolved), op=MPI.SUM) > 0:
        raise RuntimeError(
            f"`mapping_function` did not reach a vertex outside `indicator` within"
            f" {max_chain_length} applications. Either the two functions disagree, or the"
            " mapping cycles: an indicator vertex is mapped onto another that maps back."
        )

    # For each vertex that will be replaced, find which process should take it over
    vertex_ownership_data = dolfinx.geometry.determine_point_ownership(
        mesh, mapped_vertex_coords, padding=eps
    )
    # On process that has taken over a vertex, find the closest vertex (local to proc) that
    # will be its replacement
    acquired_vertex_coords = vertex_ownership_data.dest_points
    potential_closest_vertices = dolfinx.mesh.compute_incident_entities(
        mesh.topology, vertex_ownership_data.dest_cells, mesh.topology.dim, 0
    )
    closest_vertex_bb_tree = dolfinx.geometry.bb_tree(
        mesh, 0, entities=potential_closest_vertices, padding=eps
    )
    closest_vertex_mid_tree = dolfinx.geometry.create_midpoint_tree(
        mesh, 0, potential_closest_vertices
    )
    closest_vertex = dolfinx.geometry.compute_closest_entity(
        closest_vertex_bb_tree,
        closest_vertex_mid_tree,
        mesh,
        acquired_vertex_coords,
    )

    # `compute_closest_entity` returns the closest candidate whether or not it is anywhere
    # near the query point, so check that the mapped point really landed on it. A mapping
    # wrong by a whole cell otherwise gives a valid-looking mesh that is not periodic.
    closest_vertex_coords = mesh.geometry.x[
        dolfinx.mesh.entities_to_geometry(mesh, 0, closest_vertex).reshape(-1)
    ]
    snap_distance = np.linalg.norm(
        closest_vertex_coords - acquired_vertex_coords, axis=1
    )
    num_unsnapped = int(np.count_nonzero(snap_distance > eps))
    # Collective: the condition is reduced so that either every process raises or none
    # does. A one-sided raise would leave the others blocked in the rebuild's exchanges.
    total_unsnapped = comm.allreduce(num_unsnapped, op=MPI.SUM)
    if total_unsnapped > 0:
        worst = comm.allreduce(float(np.max(snap_distance, initial=0.0)), op=MPI.MAX)
        raise RuntimeError(
            f"`mapping_function` did not map {total_unsnapped} vertices onto a vertex of"
            f" the mesh; the largest gap between a mapped point and the closest vertex is"
            f" {worst:.3e}. Every mapped point has to land on the vertex it is meant to be"
            " identified with, otherwise the result is a mesh with the expected vertex and"
            " cell counts that is not periodic."
        )

    return VertexCorrespondence(
        indicator_vertices=indicator_vertices,
        indicator_facets=indicator_facets,
        src_owner=vertex_ownership_data.src_owner,
        dest_owner=vertex_ownership_data.dest_owner,
        partner_vertex=closest_vertex,
    )


def _reduced_vertex_map(mesh, indicator_vertices):
    """The vertex index map of `mesh` with the given vertices removed.

    Collective.

    Args:
        mesh: The mesh whose vertex map is to be reduced.
        indicator_vertices: Local vertices to leave out of the reduced map.

    Returns:
        ``(sub_map, parent_to_sub)``: the reduced index map, and the map from a local
        vertex of `mesh` to its local index in `sub_map`, ``-1`` at the removed vertices.

    Raises:
        RuntimeError: If removing the vertices would move ownership of a vertex that is
            kept. The reduced map is only usable here while ownership is unchanged.
    """
    vertex_map = mesh.topology.index_map(0)
    num_vertices_local = vertex_map.size_local + vertex_map.num_ghosts
    keep_vertices = np.ones(num_vertices_local, dtype=np.bool_)
    keep_vertices[indicator_vertices] = False
    reduced_vertices = np.flatnonzero(keep_vertices)
    # Compat: 0.12 moved `create_sub_index_map` to `dolfinx.common` and made it report
    # ownership changes instead of taking a flag. Once only that form is supported the
    # branch goes away and this function is the four lines around the call.
    if hasattr(dolfinx.common, "create_sub_index_map"):
        sub_map, sub_to_parent, changed_owner = dolfinx.common.create_sub_index_map(
            vertex_map, reduced_vertices
        )
        if vertex_map.comm.allreduce(changed_owner, op=MPI.LOR):
            raise RuntimeError(
                "Vertex ownership has changed, which is not supported. "
                "Please report this issue to the dolfinx developers."
            )
    else:
        sub_map, sub_to_parent = dolfinx.cpp.common.create_sub_index_map(
            vertex_map, reduced_vertices, allow_owner_change=False
        )

    parent_to_sub = np.full(num_vertices_local, -1, dtype=np.int32)
    parent_to_sub[sub_to_parent] = np.arange(sub_to_parent.size, dtype=np.int32)
    return sub_map, parent_to_sub


def _partner_in_reduced_map(comm, sub_map, parent_to_sub, partner_vertex):
    """Look up local vertices in a reduced vertex map, as global index and owner.

    Collective, so that a vertex missing from the map raises on every process rather than
    on one.

    Args:
        comm: The communicator to reduce the check over.
        sub_map: A reduced vertex map, from :func:`_reduced_vertex_map`.
        parent_to_sub: Its companion map from local vertices, as returned alongside it.
        partner_vertex: Local vertices to look up.

    Returns:
        ``(global_vertices, owners)``, both indexed like `partner_vertex`: each vertex's
        global index in `sub_map`, and the rank that owns it.

    Raises:
        AssertionError: If any of `partner_vertex` is one of the vertices `sub_map` was
            built without.
    """
    sub_partner_vertex = parent_to_sub[partner_vertex]

    # Collective, so that either every process raises or none does. A one-sided raise
    # would leave the others blocked in the exchanges that follow.
    num_missing_replacements = int(np.count_nonzero(sub_partner_vertex == -1))
    if comm.allreduce(num_missing_replacements, op=MPI.SUM) > 0:
        raise AssertionError(
            "Partner vertex not in submap: a vertex was paired with another vertex that"
            " is itself marked for replacement, so the replacement has been removed too."
            " Under periodicity in several directions the pairing has to carry a corner"
            " vertex all the way to its root, applying every offset that applies to it."
        )

    global_vertices = sub_map.local_to_global(sub_partner_vertex).astype(np.int64)
    owners = get_ownership(sub_map)[sub_partner_vertex].copy()
    return global_vertices, owners


def _owners_from_source(vertex_sources, recv_num_cells, recv_potential_cell_owners):
    """For each sender, the owners of the cells it sent, together with the sender.

    Reads the answer off data already received, so it communicates nothing.

    Args:
        vertex_sources: The sending ranks, ascending and without repeats.
        recv_num_cells: How many cells were received from each of them, in the same order.
        recv_potential_cell_owners: Owner of each received cell, grouped by sender, so of
            length ``recv_num_cells.sum()``.

    Returns:
        A rank from `vertex_sources` to that union, ascending and without repeats. The
        caller uses it as the set of ranks to reach on that sender's behalf, which is
        wider than the sender alone because several of them may hold a cell at a vertex.
    """
    source_offsets = np.zeros(len(recv_num_cells) + 1, dtype=np.int64)
    np.cumsum(recv_num_cells, out=source_offsets[1:])
    return {
        int(src): np.union1d(
            recv_potential_cell_owners[source_offsets[j] : source_offsets[j + 1]],
            np.array([src], dtype=np.int32),
        ).astype(np.int32)
        for j, src in enumerate(vertex_sources)
    }


def _seam_facet_destinations(
    mesh, indicator_vertices, indicator_facets, src_owner, owners_from_source
):
    """Expand each facet into one pair per rank reachable through its vertices.

    A facet's vertices name ranks in `src_owner`, each of which `owners_from_source`
    widens to a set; the facet is paired with the union over its vertices.

    Collective, so that a facet whose vertices are not all in `indicator_vertices` raises
    on every process rather than on one.

    Args:
        mesh: The mesh `indicator_facets` and `indicator_vertices` are local to.
        indicator_vertices: Local vertices that `src_owner` is keyed on, in that order.
        indicator_facets: Local facets to expand.
        src_owner: One rank per entry of `indicator_vertices`.
        owners_from_source: A rank to the ranks it stands for, from
            :func:`_owners_from_source`.

    Returns:
        ``(facet, rank)`` pairs as an ``(n, 2)`` array, de-duplicated and sorted
        lexicographically, where `facet` indexes into `indicator_facets`. A facet appears
        once per distinct rank, so the array is longer than `indicator_facets`.

    Raises:
        RuntimeError: If a facet in `indicator_facets` has a vertex that is not in
            `indicator_vertices`.
    """
    comm = mesh.comm
    mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
    f_to_v = mesh.topology.connectivity(mesh.topology.dim - 1, 0)

    vertex_map = mesh.topology.index_map(0)
    # position of a local vertex in `indicator_vertices`, which is the index that
    # `src_owner` and `global_replacement_*` are keyed on: entry k describes the k-th
    # local vertex that will be replaced.
    position_in_indicator_vertices = np.full(
        vertex_map.size_local + vertex_map.num_ghosts, -1, dtype=np.int32
    )
    position_in_indicator_vertices[indicator_vertices] = np.arange(
        len(indicator_vertices), dtype=np.int32
    )

    # NOTE: assumes every facet has the same number of vertices. That holds for a single
    # cell type but not for a mixed-topology mesh (a prism has both triangular and
    # quadrilateral facets). To support it, walk `f_to_v.offsets` facet by facet and build
    # the pairs from a ragged array instead of a rectangular one. Nothing downstream has
    # to change: every array from here on carries one entry per pair, never one per facet.
    num_facet_vertices = int(f_to_v.offsets[1] - f_to_v.offsets[0])
    # Equivalent to:
    #     for i, f in enumerate(indicator_facets):
    #         for j in range(num_facet_vertices):
    #             facet_vertices[i, j] = f_to_v.links(f)[j]
    facet_vertices = f_to_v.array[
        (
            f_to_v.offsets[indicator_facets][:, None]
            + np.arange(num_facet_vertices, dtype=np.int32)
        ).reshape(-1)
    ].reshape(len(indicator_facets), num_facet_vertices)
    facet_vertex_positions = position_in_indicator_vertices[facet_vertices]
    num_unmarked = int(np.count_nonzero(facet_vertex_positions == -1))
    if comm.allreduce(num_unmarked, op=MPI.SUM) > 0:
        raise RuntimeError(
            "A facet on the seam has a vertex that is not marked for replacement."
            " `indicator_facets` and `indicator_vertices` have to agree for the facet to"
            " be identified with another one."
        )

    # one (facet, rank) pair per distinct destination among the facet's vertices.
    # Equivalent to:
    #     pairs = set()
    #     for i in range(len(indicator_facets)):
    #         for j in range(num_facet_vertices):
    #             src = src_owner[facet_vertex_positions[i, j]]
    #             for rank in owners_from_source[src]:
    #                 pairs.add((i, rank))
    #     facet_pairs = sorted(pairs)
    # `np.unique(..., axis=0)` both de-duplicates and sorts lexicographically by
    # (facet, rank), which is the order the packing downstream expects.
    facet_vertex_source = src_owner[facet_vertex_positions]
    pair_facet, pair_rank = [], []
    for src, owners in owners_from_source.items():
        hit_facet = np.nonzero(facet_vertex_source == src)[0].astype(np.int32)
        if hit_facet.size and owners.size:
            pair_facet.append(np.repeat(hit_facet, owners.size))
            pair_rank.append(np.tile(owners, hit_facet.size))
    if not pair_facet:
        return np.zeros((0, 2), dtype=np.int32)
    return np.unique(
        np.stack([np.concatenate(pair_facet), np.concatenate(pair_rank)], axis=1),
        axis=0,
    )


def _number_new_ghosts(index_map, global_indices, *payloads):
    """Translate global indices to local, numbering the unknown ones as further ghosts.

    An index `index_map` holds keeps its local number; one it does not is given the next
    number after the map's existing ghosts. `index_map` itself is not modified -- the
    caller builds the replacement from `new_ghosts` and the gathered owners -- but the
    numbering assumes the new ghosts are appended to it in the order returned.

    Local; no communication.

    Args:
        index_map: The map to translate against.
        global_indices: Flat array of global indices, repeats allowed.
        payloads: Arrays indexed like `global_indices`, each carrying one value per entry.
            All occurrences of an index must carry the same value, which is asserted.

    Returns:
        ``(local, new_ghosts, first_occurrence, gathered)``: `global_indices` in local
        numbering; the global indices of the new ghosts, ascending; the position in
        `global_indices` where each of them first occurs, so that an array not available
        yet can be reduced later with ``values[first_occurrence]``; and `payloads` so
        reduced, as a tuple.
    """
    local = index_map.global_to_local(global_indices)
    missing = np.flatnonzero(local == -1)
    new_ghosts, pos, inverse = np.unique(
        global_indices[missing], return_index=True, return_inverse=True
    )
    first = index_map.size_local + index_map.num_ghosts
    local[missing] = (first + np.arange(len(new_ghosts), dtype=np.int32))[inverse]

    gathered = []
    for values in payloads:
        occurrences = np.asarray(values)[missing]
        chosen = occurrences[pos]
        assert np.array_equal(occurrences, chosen[inverse]), (
            "occurrences of the same global index disagree on an accompanying value"
        )
        gathered.append(chosen)
    return local, new_ghosts, missing[pos], tuple(gathered)


def _build_periodic_mesh(
    mesh, correspondence: VertexCorrespondence
) -> tuple[dolfinx.mesh.Mesh, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Rebuild `mesh` with the vertex pairs of `correspondence` identified.

    Purely topological: the correspondence already says which vertex replaces which and
    which ranks are involved, so nothing here evaluates a coordinate or a user function.
    See {py:func}`create_periodic_mesh` for the return value.
    """
    indicator_vertices = correspondence.indicator_vertices
    indicator_facets = correspondence.indicator_facets
    src_owner = correspondence.src_owner
    dest_owner = correspondence.dest_owner
    partner_vertex = correspondence.partner_vertex

    comm = mesh.comm
    geometry = mesh.geometry._cpp_object
    topology = mesh.topology
    num_vertices = dolfinx.cpp.mesh.cell_num_vertices(mesh.topology.cell_type)
    assert len(mesh.geometry.dofmaps) == 1, "Only one geometry dofmap is supported"
    num_nodes = mesh.geometry.dofmaps[0].shape[1]

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # The mesh without the vertices that are being replaced, and the partners resolved
    # against it: everything below is expressed in this reduced numbering.
    sub_map_without_ghosts, parent_to_sub = _reduced_vertex_map(
        mesh, indicator_vertices
    )
    global_vertices, send_vertex_owner = _partner_in_reduced_map(
        comm, sub_map_without_ghosts, parent_to_sub, partner_vertex
    )

    # Every boundary facet the process knows of, so it has to be broadened:
    # `exterior_facet_indices` returns owned facets only, and a process that merely ghosts
    # a boundary facet at a replacement vertex would never ship the cell behind it,
    # leaving a seam facet with one cell instead of two on the far side.
    mesh.topology.create_entities(mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    org_mesh_ext_facets = broadcast_marked_entities(
        mesh,
        mesh.topology.dim - 1,
        dolfinx.mesh.exterior_facet_indices(mesh.topology),
    )
    mesh.topology.create_connectivity(0, mesh.topology.dim - 1)

    # Get vertex and geometry dofs to send
    assert len(mesh.geometry.dofmaps) == 1, "Only one geometry dofmap is supported"
    geom_dm = mesh.geometry.dofmaps[0]
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)

    # Pack data from process taking over vertex to process that has lost vertex
    assert np.all(dest_owner[:-1] <= dest_owner[1:]), "Vertex owners are not sorted"
    vertex_destinations, send_vertices_per_proc = np.unique(
        dest_owner, return_counts=True
    )
    num_cells_per_proc = np.zeros_like(
        send_vertices_per_proc, dtype=np.int32
    )  # Each vertex might be connected to multiple cells
    # Packing offset
    offsets = np.zeros(len(send_vertices_per_proc) + 1, dtype=np.int32)
    np.cumsum(send_vertices_per_proc, out=offsets[1:])

    # For each replacement vertex, find all facets that are connected to the vertex and is on the boundary.
    # Then, get each cell connected to this facet.
    # Seeded with an empty array so the concatenations below need no special case for a
    # process that sends to nobody.
    send_ghost_cells_from_new_owner = [np.zeros(0, dtype=np.int32)]
    new_cell_topology_dm = [np.zeros(0, dtype=np.int32)]
    for i in range(len(send_vertices_per_proc)):
        _vertices = partner_vertex[
            offsets[i] : offsets[i + 1]
        ]  # Works because dest owners are sorted
        connected_facets = dolfinx.mesh.compute_incident_entities(
            mesh.topology, _vertices, 0, mesh.topology.dim - 1
        )
        con_ext_facets = np.intersect1d(connected_facets, org_mesh_ext_facets)
        con_ext_cells = dolfinx.mesh.compute_incident_entities(
            mesh.topology, con_ext_facets, mesh.topology.dim - 1, mesh.topology.dim
        )
        # Equivalent to:
        #     for c in con_ext_cells:
        #         send_ghost_cells_from_new_owner.push_back(c);
        #         num_cells_per_proc[i] += 1;
        #         for (int j = 0; j < num_vertices; ++j)
        #             new_cell_topology_dm.push_back(c_to_v.links(c)[j]);
        # `con_ext_cells` is already sorted and unique, so the gather preserves the order
        # the loop would append in. As elsewhere, this assumes every cell has the same
        # number of vertices; the reshape further down assumes it too.
        num_cells_per_proc[i] = len(con_ext_cells)
        send_ghost_cells_from_new_owner.append(con_ext_cells)
        new_cell_topology_dm.append(
            c_to_v.array[
                (
                    c_to_v.offsets[con_ext_cells][:, None]
                    + np.arange(num_vertices, dtype=np.int32)
                ).reshape(-1)
            ]
        )
    send_ghost_cells_from_new_owner = np.concatenate(
        send_ghost_cells_from_new_owner
    ).astype(np.int32)
    new_cell_topology_dm = np.concatenate(new_cell_topology_dm).astype(np.int32)

    # Create new owner to old owner communicator
    vertex_sources, recv_vertices_per_proc = np.unique(src_owner, return_counts=True)
    new_owner_to_old_comm = comm.Create_dist_graph_adjacent(
        vertex_sources, vertex_destinations, reorder=False
    )

    # Send replacement vertices to process that has lost vertex
    recv_replacement_vertices = np.empty(recv_vertices_per_proc.sum(), dtype=np.int64)
    all_to_allv(
        new_owner_to_old_comm,
        global_vertices,
        send_vertices_per_proc,
        recv_replacement_vertices,
        recv_vertices_per_proc,
    )

    # Send owner of said vertex to the process that will use it as a replacement
    recv_replacement_owner = np.empty(recv_vertices_per_proc.sum(), dtype=np.int32)
    all_to_allv(
        new_owner_to_old_comm,
        send_vertex_owner,
        send_vertices_per_proc,
        recv_replacement_owner,
        recv_vertices_per_proc,
    )

    # For the data that will be received, the received has to be
    # ordered by their initial position in `indicator_vertices`, not by src_rank
    current_rank_to_recv = compute_insert_position(
        src_owner, vertex_sources, recv_vertices_per_proc
    )
    # Invert map so that we can insert the data
    dest_ranks_to_current = np.zeros(len(indicator_vertices), dtype=np.int64)
    dest_ranks_to_current[current_rank_to_recv] = np.arange(
        len(dest_ranks_to_current), dtype=np.int32
    )

    # Global replacement index
    global_replacement_vertex = np.full(len(indicator_vertices), -1, dtype=np.int64)
    global_replacement_vertex[dest_ranks_to_current] = recv_replacement_vertices
    global_replacement_owner = np.full(len(indicator_vertices), -1, dtype=np.int64)
    global_replacement_owner[dest_ranks_to_current] = recv_replacement_owner
    # Collective, and explicit about the cause: a -1 here means the mapped point was not
    # found in any cell of the mesh, i.e. `mapping_function` moved it outside the domain.
    num_unowned = int(
        np.count_nonzero(
            (global_replacement_vertex == -1) | (global_replacement_owner == -1)
        )
    )
    if comm.allreduce(num_unowned, op=MPI.SUM) > 0:
        raise RuntimeError(
            f"{comm.allreduce(num_unowned, op=MPI.SUM)} mapped vertices were not found in"
            " any cell of the mesh. `mapping_function` has to land inside the domain; a"
            " point that leaves it has no owner and no replacement vertex."
        )
    # print(MPI.COMM_WORLD.rank, global_replacement_vertex, global_replacement_owner)
    # Set up ownership structure of cells, nodes and vertices on the process
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    cell_owners = get_ownership(cell_map)
    assert (send_ghost_cells_from_new_owner > -1).all()
    assert (
        send_ghost_cells_from_new_owner < cell_map.size_local + cell_map.num_ghosts
    ).all()
    global_ghost_cells_from_new_owner = cell_map.local_to_global(
        np.array(send_ghost_cells_from_new_owner, dtype=np.int32)
    ).astype(np.int64)

    # Map to global indices

    vertex_owners = get_ownership(sub_map_without_ghosts)
    subdofmap_for_new_owner_ghost_cells_local = parent_to_sub[new_cell_topology_dm]
    replacement_positions = subdofmap_for_new_owner_ghost_cells_local == -1
    unmodified_positions = np.invert(replacement_positions)
    gl_new_cell_topology_dm = np.full_like(
        subdofmap_for_new_owner_ghost_cells_local, -1, dtype=np.int64
    )
    gl_new_cell_topology_dm[unmodified_positions] = (
        sub_map_without_ghosts.local_to_global(
            subdofmap_for_new_owner_ghost_cells_local[unmodified_positions]
        ).astype(np.int64)
    )
    gl_new_cell_topology_owners = np.full_like(
        gl_new_cell_topology_dm, -1, dtype=np.int32
    )
    gl_new_cell_topology_owners[unmodified_positions] = vertex_owners[
        subdofmap_for_new_owner_ghost_cells_local[unmodified_positions]
    ]

    # Replace vertices that has been removed by their new global index
    relative_replacement_pos = find_position(
        new_cell_topology_dm[replacement_positions], indicator_vertices
    )
    gl_new_cell_topology_dm[replacement_positions] = global_replacement_vertex[
        relative_replacement_pos
    ]
    gl_new_cell_topology_owners[replacement_positions] = global_replacement_owner[
        relative_replacement_pos
    ]
    assert (gl_new_cell_topology_owners != -1).all()
    assert (gl_new_cell_topology_dm != -1).all()

    # Compute number of cells to send and receive
    recv_num_cells = np.zeros_like(recv_vertices_per_proc, dtype=np.int32)
    all_to_all(new_owner_to_old_comm, num_cells_per_proc, recv_num_cells)

    # Send cells and owners to process that lost vertex
    recv_potential_ghost_cells = np.empty(recv_num_cells.sum(), dtype=np.int64)
    all_to_allv(
        new_owner_to_old_comm,
        global_ghost_cells_from_new_owner,
        num_cells_per_proc,
        recv_potential_ghost_cells,
        recv_num_cells,
    )
    recv_potential_cell_owners = np.empty(recv_num_cells.sum(), dtype=np.int32)
    send_ghost_cell_owners = cell_owners[send_ghost_cells_from_new_owner].copy()
    all_to_allv(
        new_owner_to_old_comm,
        send_ghost_cell_owners,
        num_cells_per_proc,
        recv_potential_cell_owners,
        recv_num_cells,
    )

    # Send oci
    recv_potential_cell_oci = np.empty(recv_num_cells.sum(), dtype=np.int64)
    send_cell_oci = (
        mesh.topology.original_cell_index[send_ghost_cells_from_new_owner]
        .copy()
        .astype(np.int64)
    )
    all_to_allv(
        new_owner_to_old_comm,
        send_cell_oci,
        num_cells_per_proc,
        recv_potential_cell_oci,
        recv_num_cells,
    )

    # Check if received cells are already in cell map
    potential_ghosts_as_local = cell_map.global_to_local(recv_potential_ghost_cells)
    cell_filter = np.flatnonzero(potential_ghosts_as_local == -1)

    new_cells_from_new_vertex_owner, vertex_owner_cell_position = np.unique(
        recv_potential_ghost_cells[cell_filter], return_index=True
    )
    # Send dofmaps for topology
    new_top_dm_on_proc = np.empty((recv_num_cells.sum(), num_vertices), dtype=np.int64)
    all_to_allv(
        new_owner_to_old_comm,
        gl_new_cell_topology_dm,
        num_vertices * num_cells_per_proc,
        new_top_dm_on_proc,
        num_vertices * recv_num_cells,
    )

    # Send ownership of vertices
    top_dm_ownership = np.empty_like(new_top_dm_on_proc, dtype=np.int32)
    all_to_allv(
        new_owner_to_old_comm,
        gl_new_cell_topology_owners,
        num_vertices * num_cells_per_proc,
        top_dm_ownership,
        num_vertices * recv_num_cells,
    )

    # Compute the vertex ghosts.
    #
    # The incoming cells are described by global vertex indices in the reduced (submap)
    # numbering. Some of those vertices this process already holds; the rest have to
    # become ghosts, and the index map has to be extended with them before the incoming
    # dofmap can be expressed locally at all. `cell_filter` has already dropped the cells
    # that are not actually new, so only their vertices are in question here.
    # `cell_filter` has already dropped the cells that are not actually new, so only the
    # vertices of the genuinely new ones are in question here.
    local_dm, new_ghost_vertices, _, (new_ghost_owners,) = _number_new_ghosts(
        sub_map_without_ghosts,
        new_top_dm_on_proc[cell_filter].reshape(-1),
        top_dm_ownership[cell_filter].reshape(-1),
    )
    new_local_size = int(sub_map_without_ghosts.size_local)

    # The ghost list of the index map that supersedes the submap below. The order matches
    # the numbering just assigned: existing ghosts first, then the new ones.
    new_ghosts = np.hstack([sub_map_without_ghosts.ghosts, new_ghost_vertices]).astype(
        np.int64
    )
    new_owners = np.hstack([sub_map_without_ghosts.owners, new_ghost_owners]).astype(
        np.int32
    )
    # A ghost is owned elsewhere by definition. A self-owned entry here would mean
    # `global_to_local` failed to find a vertex this process does in fact own.
    assert (new_owners != comm.rank).all()

    # Check if index is already in (reduced) vertex map
    local_replacement_vertex = sub_map_without_ghosts.global_to_local(
        global_replacement_vertex
    )
    is_local_indicator = local_replacement_vertex != -1
    existing_vertices = np.flatnonzero(is_local_indicator)

    # Vertex map is temporary, as we need to extend it with additional ghosts on the process taking over facets
    tmp_vertex_map = _compat_index_map(
        comm, new_local_size, new_ghosts, new_owners, tag=1102
    )
    tmp_vertex_ownership = get_ownership(tmp_vertex_map)

    # Create replacement map
    replacement_map = parent_to_sub.copy()
    # Replace existing vertices
    replacement_map[indicator_vertices[existing_vertices]] = local_replacement_vertex[
        existing_vertices
    ]

    # For new ghosts, add the to replacement map
    is_new_replacement = np.invert(is_local_indicator)
    replacement_ghosts = global_replacement_vertex[is_new_replacement]
    assert np.isin(replacement_ghosts, new_ghosts).all(), (
        "Replacement ghost not in new ghost list"
    )
    if len(replacement_ghosts) > 0:
        local_replacement_position = find_position(replacement_ghosts, new_ghosts)

        replacement_map[indicator_vertices[is_new_replacement]] = (
            new_local_size + local_replacement_position
        )

    geom_im = mesh.geometry.index_map()
    node_owners = get_ownership(geom_im)

    # Convert old vertex_to_dofmap to reduced set
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    new_c = replacement_map[c_to_v.array].reshape(-1, num_vertices)
    extra_dm = local_dm.reshape(-1, num_vertices)[vertex_owner_cell_position]

    # --- 2 --- Update geometry with new (ghosted) cells

    # Extend geometry with extra cells
    new_cell_geom_dm = geom_dm[send_ghost_cells_from_new_owner]
    assert (new_cell_geom_dm > -1).all()
    assert (new_cell_geom_dm < geom_im.size_local + geom_im.num_ghosts).all()
    gl_new_cell_geom_dm = geom_im.local_to_global(new_cell_geom_dm.reshape(-1)).astype(
        np.int64
    )

    # Send potential new ghosts
    add_geom_dm = np.empty((recv_num_cells.sum(), num_nodes), dtype=np.int64)
    all_to_allv(
        new_owner_to_old_comm,
        gl_new_cell_geom_dm,
        num_nodes * num_cells_per_proc,
        add_geom_dm,
        num_nodes * recv_num_cells,
    )

    # Send owners of potential new ghost nodes
    send_geom_owners = node_owners[new_cell_geom_dm.reshape(-1)]
    add_geom_own = np.empty((recv_num_cells.sum(), num_nodes), dtype=np.int32)
    all_to_allv(
        new_owner_to_old_comm,
        send_geom_owners,
        num_nodes * num_cells_per_proc,
        add_geom_own,
        num_nodes * recv_num_cells,
    )

    # Send igi for potential new nodes
    send_igi = mesh.geometry.input_global_indices[new_cell_geom_dm.reshape(-1)].astype(
        np.int64
    )
    recv_igi = np.empty((recv_num_cells.sum(), num_nodes), dtype=np.int64)
    all_to_allv(
        new_owner_to_old_comm,
        send_igi,
        num_nodes * num_cells_per_proc,
        recv_igi,
        num_nodes * recv_num_cells,
    )

    # Compute new ghost nodes
    local_geometry_dm, new_ghost_nodes, first_new_node, (new_ghost_owners, new_igi) = (
        _number_new_ghosts(
            geom_im,
            add_geom_dm[cell_filter].flatten(),
            add_geom_own[cell_filter].flatten(),
            recv_igi[cell_filter].flatten(),
        )
    )
    num_local_nodes = geom_im.size_local

    # Communicate geometry coordinates (to process that has lost vertex)
    node_coordinates = mesh.geometry.x[new_cell_geom_dm.reshape(-1)].flatten()
    geom_coords = np.empty(
        num_nodes * 3 * recv_num_cells.sum(), dtype=mesh.geometry.x.dtype
    )
    send_coord_msg = [
        node_coordinates,
        num_nodes * 3 * num_cells_per_proc,
        mpi_dtype[mesh.geometry.x.dtype.type],
    ]
    recv_coord_msg = [
        geom_coords,
        num_nodes * 3 * recv_num_cells,
        mpi_dtype[mesh.geometry.x.dtype.type],
    ]
    new_owner_to_old_comm.Neighbor_alltoallv(send_coord_msg, recv_coord_msg)
    extra_geom_dm = local_geometry_dm.reshape(-1, num_nodes)[vertex_owner_cell_position]

    # --- 3 --- Communicate cells from process that has lost vertex to process that has taken over vertex

    # Where each seam facet's cell has to go comes from the vertex correspondence.
    # `src_owner` names a cell owner, not the owner of the replacement vertex: the rank
    # holding the cell on the far side of the seam may only ghost that vertex, and would
    # then never receive this cell. It names only one such rank, so the set is widened
    # from what phase 1 already shipped back.
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    f_to_c = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facet_pairs = _seam_facet_destinations(
        mesh,
        indicator_vertices,
        indicator_facets,
        src_owner,
        _owners_from_source(vertex_sources, recv_num_cells, recv_potential_cell_owners),
    )

    lost_facet_position = facet_pairs[:, 0]
    lost_dest_ranks = np.ascontiguousarray(facet_pairs[:, 1], dtype=np.int32)

    # Taking only the first incident cell is safe here: `indicator_facets` comes from
    # `locate_entities_boundary`, so every facet is exterior and has exactly one cell.
    # Exteriority is a global property, so the scatter that broadens the set above
    # preserves it. One entry per (facet, destination) pair, so a cell repeats once per
    # rank it must reach; everything packed below stays indexed by this array.
    cells_losing_vertex = f_to_c.array[f_to_c.offsets[indicator_facets]][
        lost_facet_position
    ]
    assert (cells_losing_vertex > -1).all()
    assert (cells_losing_vertex < cell_map.size_local + cell_map.num_ghosts).all()
    cells_losing_vertex_gl = cell_map.local_to_global(cells_losing_vertex)

    # Pack dofmap for each of these cells, replacing the vertices that are removed with mapped vertices
    renumbered_dm = new_c[cells_losing_vertex].reshape(-1)
    assert (renumbered_dm > -1).all()
    assert (renumbered_dm < tmp_vertex_map.size_local + tmp_vertex_map.num_ghosts).all()
    lost_cells_dm_global = tmp_vertex_map.local_to_global(renumbered_dm)
    lost_cells_dm_owners = tmp_vertex_ownership[renumbered_dm]

    # Pack dofmap,owners and igi of geometry, not in sorted by communication proc
    org_geom_dm_cells_losing_vertex = mesh.geometry.dofmaps[0][
        cells_losing_vertex
    ].reshape(-1)
    lost_geom_dm = geom_im.local_to_global(org_geom_dm_cells_losing_vertex)
    assert (org_geom_dm_cells_losing_vertex > -1).all()
    assert (
        org_geom_dm_cells_losing_vertex < geom_im.size_local + geom_im.num_ghosts
    ).all()

    lost_geom_owner = node_owners[org_geom_dm_cells_losing_vertex]
    lost_geom_igi = mesh.geometry.input_global_indices[org_geom_dm_cells_losing_vertex]

    # Compute insertion position based on cell ownership
    lost_src_ranks, num_send_lost_cells = np.unique(lost_dest_ranks, return_counts=True)
    num_send_lost_cells = num_send_lost_cells.astype(np.int32)
    lost_cell_insert_pos = compute_insert_position(
        lost_dest_ranks, lost_src_ranks, num_send_lost_cells
    )

    # Pack cells data
    lost_cells_send_buffer = np.empty(len(cells_losing_vertex), dtype=np.int64)
    lost_cells_send_buffer[lost_cell_insert_pos] = cells_losing_vertex_gl
    lost_owners_send_buffer = np.empty_like(lost_cells_send_buffer, dtype=np.int32)
    lost_owners_send_buffer[lost_cell_insert_pos] = cell_owners[cells_losing_vertex]
    lost_oci_send_buffer = np.empty_like(lost_cells_send_buffer, dtype=np.int64)
    lost_oci_send_buffer[lost_cell_insert_pos] = mesh.topology.original_cell_index[
        cells_losing_vertex
    ]

    # Pack topology data
    lost_insert_pos_top_dm = unroll_insert_position(lost_cell_insert_pos, num_vertices)
    lost_cells_dofmap_send_buffer = np.empty_like(
        lost_insert_pos_top_dm, dtype=np.int64
    )
    lost_cells_dofmap_send_buffer[lost_insert_pos_top_dm] = lost_cells_dm_global
    lost_cells_dofmap_owners_buffer = np.empty_like(
        lost_insert_pos_top_dm, dtype=np.int32
    )
    lost_cells_dofmap_owners_buffer[lost_insert_pos_top_dm] = lost_cells_dm_owners

    lost_insert_pos_geom_dm = unroll_insert_position(lost_cell_insert_pos, num_nodes)
    lost_cells_gdofmap_send_buffer = np.empty_like(
        lost_insert_pos_geom_dm, dtype=np.int64
    )
    lost_cells_gdofmap_send_buffer[lost_insert_pos_geom_dm] = lost_geom_dm
    lost_cells_gdofmap_owner_buffer = np.empty_like(
        lost_insert_pos_geom_dm, dtype=np.int32
    )
    lost_cells_gdofmap_owner_buffer[lost_insert_pos_geom_dm] = lost_geom_owner
    lost_cells_gdofmap_igi_buffer = np.empty_like(
        lost_insert_pos_geom_dm, dtype=np.int64
    )
    lost_cells_gdofmap_igi_buffer[lost_insert_pos_geom_dm] = lost_geom_igi

    xtype = mesh.geometry.x.dtype
    lost_insert_pos_geom_coord = unroll_insert_position(
        lost_cell_insert_pos, 3 * num_nodes
    )
    lost_geom_coords = mesh.geometry.x[org_geom_dm_cells_losing_vertex].flatten()
    lost_cells_coords_buffer = np.empty_like(lost_insert_pos_geom_coord, dtype=xtype)
    lost_cells_coords_buffer[lost_insert_pos_geom_coord] = lost_geom_coords

    # Create communicator
    # `Create_dist_graph` only needs the edges leaving this process; MPI works out who
    # sends to me, so no separate discovery exchange is needed. The counts per source then
    # come from one neighborhood exchange on that graph.
    assert isinstance(comm, MPI.Intracomm)
    lost_cells_to_gainer_comm = comm.Create_dist_graph(
        [comm.rank], [len(lost_src_ranks)], lost_src_ranks.tolist(), reorder=False
    )
    recv_lost_cells_ranks, sent_to_ranks, _ = (
        lost_cells_to_gainer_comm.Get_dist_neighbors()
    )
    recv_lost_cells_ranks = np.asarray(recv_lost_cells_ranks, dtype=np.int32)
    assert np.array_equal(np.asarray(sent_to_ranks, dtype=np.int32), lost_src_ranks)
    num_recv_lost_cells = np.zeros(len(recv_lost_cells_ranks), dtype=np.int32)
    all_to_all(lost_cells_to_gainer_comm, num_send_lost_cells, num_recv_lost_cells)

    total_recv_lost_cells = num_recv_lost_cells.sum()
    lost_cells_recv_buffer = np.empty(total_recv_lost_cells, dtype=np.int64)

    # Communicate cells
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_cells_send_buffer,
        num_send_lost_cells,
        lost_cells_recv_buffer,
        num_recv_lost_cells,
    )

    # Communicate owners of potential new ghost cells
    lost_cells_owners_recv_buffer = np.empty_like(
        lost_cells_recv_buffer, dtype=np.int32
    )
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_owners_send_buffer,
        num_send_lost_cells,
        lost_cells_owners_recv_buffer,
        num_recv_lost_cells,
    )

    # Communicate oci of potential new ghost cells
    lost_cells_oci_recv_buffer = np.empty_like(lost_cells_recv_buffer, dtype=np.int64)
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_oci_send_buffer,
        num_send_lost_cells,
        lost_cells_oci_recv_buffer,
        num_recv_lost_cells,
    )

    # Communicate dofmap and ownership info
    lost_cells_dm_recv_buffer = np.empty(
        (total_recv_lost_cells, num_vertices), dtype=np.int64
    )
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_cells_dofmap_send_buffer,
        num_send_lost_cells * num_vertices,
        lost_cells_dm_recv_buffer,
        num_recv_lost_cells * num_vertices,
    )
    lost_cells_dm_owner_recv_buffer = np.empty_like(
        lost_cells_dm_recv_buffer, dtype=np.int32
    )
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_cells_dofmap_owners_buffer,
        num_send_lost_cells * num_vertices,
        lost_cells_dm_owner_recv_buffer,
        num_recv_lost_cells * num_vertices,
    )

    # Communicate geometry dofmap, igi, owners and coordinates
    lost_cells_gdofmap_recv_buffer = np.empty(
        (total_recv_lost_cells, num_nodes), dtype=np.int64
    )
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_cells_gdofmap_send_buffer,
        num_send_lost_cells * num_nodes,
        lost_cells_gdofmap_recv_buffer,
        num_recv_lost_cells * num_nodes,
    )
    lost_cells_gdofmap_owner_recv_buffer = np.empty_like(
        lost_cells_gdofmap_recv_buffer, dtype=np.int32
    )
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_cells_gdofmap_owner_buffer,
        num_send_lost_cells * num_nodes,
        lost_cells_gdofmap_owner_recv_buffer,
        num_recv_lost_cells * num_nodes,
    )
    lost_cells_igi_recv_buffer = np.empty_like(
        lost_cells_gdofmap_recv_buffer, dtype=np.int64
    )
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_cells_gdofmap_igi_buffer,
        num_send_lost_cells * num_nodes,
        lost_cells_igi_recv_buffer,
        num_recv_lost_cells * num_nodes,
    )

    lost_cells_node_coords_recv_buffer = np.empty(
        (total_recv_lost_cells, num_nodes, 3), dtype=mesh.geometry.x.dtype
    )
    all_to_allv(
        lost_cells_to_gainer_comm,
        lost_cells_coords_buffer,
        num_send_lost_cells * num_nodes * 3,
        lost_cells_node_coords_recv_buffer,
        num_recv_lost_cells * num_nodes * 3,
    )

    # Only add cells that are new on the process and only add them once
    lost_cell_indicator = np.flatnonzero(
        cell_map.global_to_local(lost_cells_recv_buffer) == -1
    )
    duplicate_indicator = np.isin(
        lost_cells_recv_buffer, new_cells_from_new_vertex_owner, invert=True
    )
    other_indicator = np.flatnonzero(duplicate_indicator)
    new_lost_cells_indicator = np.intersect1d(lost_cell_indicator, other_indicator)

    unique_lost_cells, unique_lost_cells_position = np.unique(
        lost_cells_recv_buffer[new_lost_cells_indicator], return_index=True
    )

    unique_lost_cells_owners = lost_cells_owners_recv_buffer[new_lost_cells_indicator][
        unique_lost_cells_position
    ]
    unique_lost_cells_oci = lost_cells_oci_recv_buffer[new_lost_cells_indicator][
        unique_lost_cells_position
    ]
    # Get dofmap in local indices
    unique_lost_cells_dm = lost_cells_dm_recv_buffer[new_lost_cells_indicator][
        unique_lost_cells_position
    ].reshape(-1)
    unique_lost_cells_dm_owners = lost_cells_dm_owner_recv_buffer[
        new_lost_cells_indicator
    ][unique_lost_cells_position].reshape(-1)
    lost_cells_dofs_as_local = tmp_vertex_map.global_to_local(unique_lost_cells_dm)

    # Find those vertex dofs that are new, and compute their new local vertex number
    lost_cells_new_vertices = np.flatnonzero(lost_cells_dofs_as_local == -1)
    (
        lost_cells_unique_new_ghosts,
        unique_ghosts_position,
        unique_ghosts_to_new_vertices,
    ) = np.unique(
        unique_lost_cells_dm[lost_cells_new_vertices],
        return_index=True,
        return_inverse=True,
    )
    lost_cells_ghost_owners = unique_lost_cells_dm_owners[lost_cells_new_vertices][
        unique_ghosts_position
    ]
    lost_cells_ghost_insert_position = (
        tmp_vertex_map.size_local + tmp_vertex_map.num_ghosts
    )
    lost_cells_dofs_as_local[lost_cells_new_vertices] = (
        lost_cells_ghost_insert_position
        + np.arange(len(lost_cells_unique_new_ghosts), dtype=np.int32)
    )[unique_ghosts_to_new_vertices]
    assert len(np.intersect1d(tmp_vertex_map.ghosts, lost_cells_unique_new_ghosts)) == 0

    # Compute ghost nodes for cells that are sent from process losing a facet
    filtered_geometry_dm = lost_cells_gdofmap_recv_buffer[new_lost_cells_indicator][
        unique_lost_cells_position
    ].flatten()
    ext_geometry_dm = geom_im.global_to_local(filtered_geometry_dm)
    new_ext_nodes = np.flatnonzero(ext_geometry_dm == -1)
    ext_gm_ghosts, extg_pos, extg_inverse_map = np.unique(
        filtered_geometry_dm[new_ext_nodes], return_index=True, return_inverse=True
    )
    filtered_geometry_o = lost_cells_gdofmap_owner_recv_buffer[
        new_lost_cells_indicator
    ][unique_lost_cells_position].flatten()
    ext_ghost_owners = filtered_geometry_o[new_ext_nodes][extg_pos]
    ext_node_pos = num_local_nodes + geom_im.num_ghosts + len(new_ghost_nodes)
    ext_geometry_dm[new_ext_nodes] = (
        ext_node_pos + np.arange(len(ext_gm_ghosts), dtype=np.int32)
    )[extg_inverse_map]
    ext_geometry_dm = ext_geometry_dm.reshape(-1, num_nodes)
    filtered_geometry_coords = lost_cells_node_coords_recv_buffer[
        new_lost_cells_indicator
    ][unique_lost_cells_position].reshape(-1, 3)[new_ext_nodes][extg_pos]
    filtered_geometry_igi = lost_cells_igi_recv_buffer[new_lost_cells_indicator][
        unique_lost_cells_position
    ].flatten()[new_ext_nodes][extg_pos]

    # --- 4 --- Convert extended topology global dofmap into local dofmap
    assert (
        len(
            np.intersect1d(
                recv_potential_ghost_cells[cell_filter][vertex_owner_cell_position],
                unique_lost_cells,
            )
        )
        == 0
    ), "Ghost in both additional maps"
    all_cell_ghosts = np.hstack(
        [cell_map.ghosts, new_cells_from_new_vertex_owner, unique_lost_cells]
    ).astype(np.int64)
    all_cell_owners = np.hstack(
        [
            cell_map.owners,
            recv_potential_cell_owners[cell_filter][vertex_owner_cell_position],
            unique_lost_cells_owners,
        ]
    ).astype(np.int32)
    assert (all_cell_owners != comm.rank).all(), "Ghosted cells on owned process"

    all_cell_oci = np.hstack(
        [
            mesh.topology.original_cell_index,
            recv_potential_cell_oci[cell_filter][vertex_owner_cell_position],
            unique_lost_cells_oci,
        ]
    ).astype(np.int64)

    assert (
        len(np.intersect1d(tmp_vertex_map.ghosts, lost_cells_unique_new_ghosts)) == 0
    ), "Ghost in both additional maps"

    all_ghosts = np.hstack(
        [tmp_vertex_map.ghosts, lost_cells_unique_new_ghosts]
    ).astype(np.int64)
    all_owners = np.hstack([tmp_vertex_map.owners, lost_cells_ghost_owners]).astype(
        np.int32
    )

    assert (all_owners != comm.rank).all(), "Ghosted vertices on owned process"

    # Create new cell and vertex map
    new_cell_map = _compat_index_map(
        comm, cell_map.size_local, all_cell_ghosts, all_cell_owners, tag=1103
    )
    new_vertex_map = _compat_index_map(
        comm, tmp_vertex_map.size_local, all_ghosts, all_owners, tag=1104
    )

    new_c_to_v = dolfinx.graph.adjacencylist(
        np.vstack([new_c, extra_dm, lost_cells_dofs_as_local.reshape(-1, num_vertices)])
    )
    new_v_to_v = dolfinx.graph.adjacencylist(
        np.arange(new_vertex_map.size_local + new_vertex_map.num_ghosts, dtype=np.int32)
    )
    assert (
        new_c_to_v.array < new_vertex_map.size_local + new_vertex_map.num_ghosts
    ).all(), "Cell to vertex map is out of bounds"

    try:
        topology = dolfinx.cpp.mesh.Topology(comm, mesh.topology.cell_type)
        topology.set_index_map(0, new_vertex_map)
        topology.set_index_map(mesh.topology.dim, new_cell_map)
        topology.set_connectivity(new_v_to_v, 0, 0)
        topology.set_connectivity(new_c_to_v, mesh.topology.dim, 0)
    except TypeError:
        try:
            topology = dolfinx.cpp.mesh.Topology(
                comm,
                mesh.topology.cell_type,
                _extract_cpp_object(new_vertex_map),
                _extract_cpp_object(new_cell_map),
                _extract_cpp_object(new_c_to_v),
                all_cell_oci,
            )
        except TypeError:
            try:
                topology = dolfinx.cpp.mesh.Topology(
                    comm,
                    mesh.topology.cell_type,
                    _extract_cpp_object(new_vertex_map),
                    _extract_cpp_object(new_cell_map),
                    _extract_cpp_object(new_c_to_v),
                    all_cell_oci,
                )
            except TypeError:
                topology = dolfinx.cpp.mesh.Topology(
                    mesh.topology.cell_type,
                    _extract_cpp_object(new_vertex_map),
                    _extract_cpp_object(new_cell_map),
                    _extract_cpp_object(new_c_to_v),
                    all_cell_oci,
                )
    c_el = dolfinx.fem.coordinate_element(
        mesh._ufl_domain.ufl_coordinate_element().basix_element
    )

    # ranges = MPI.COMM_WORLD.allgather(tmp_vertex_map.local_range)
    # for ghost, owner in zip(all_ghosts, all_owners):
    #     assert (ranges[owner][0] <= ghost) & (ghost < ranges[owner][1]), f"{comm.rank} Ghost {ghost} is not range {ranges[owner]}"
    assert (
        (all_ghosts < tmp_vertex_map.local_range[0])
        | (tmp_vertex_map.local_range[1] <= all_ghosts)
    ).all(), "Ghost "
    assert (new_vertex_map.ghosts < new_vertex_map.size_global).all(), (
        "Ghosts larger than global size"
    )

    # Create combined geometry
    extended_geom_ghosts = np.hstack(
        [geom_im.ghosts, new_ghost_nodes, ext_gm_ghosts]
    ).astype(np.int64)
    extended_geom_owners = np.hstack(
        [geom_im.owners, new_ghost_owners, ext_ghost_owners]
    ).astype(np.int32)
    # The coordinates arrive after the ghosts are numbered, so they are reduced to one
    # value per new ghost with the selection `_number_new_ghosts` handed back.
    extra_node_coords = geom_coords.reshape(-1, num_nodes, 3)[cell_filter].reshape(
        -1, 3
    )[first_new_node]

    extended_dofmap = np.vstack(
        [mesh.geometry.dofmaps[0], extra_geom_dm, ext_geometry_dm]
    ).astype(np.int32)
    extended_coords = np.vstack(
        [mesh.geometry.x, extra_node_coords, filtered_geometry_coords]
    ).astype(mesh.geometry.x.dtype)[:, : mesh.geometry.dim]
    new_node_im = _compat_index_map(
        comm, num_local_nodes, extended_geom_ghosts, extended_geom_owners, tag=1105
    )

    extended_igi = np.hstack(
        [mesh.geometry.input_global_indices, new_igi, filtered_geometry_igi]
    ).astype(np.int64)

    geometry = dolfinx.mesh.create_geometry(
        new_node_im, extended_dofmap, c_el, extended_coords, extended_igi
    )
    if mesh.geometry.x.dtype == np.float64:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float64(comm, topology, geometry._cpp_object)
    elif mesh.geometry.x.dtype == np.float32:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float32(comm, topology, geometry._cpp_object)
    else:
        raise RuntimeError(f"Unsupported dtype for mesh {mesh.geometry.x.dtype}")

    new_mesh = dolfinx.mesh.Mesh(
        cpp_mesh, domain=ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element())
    )
    new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim)
    return new_mesh, indicator_vertices, replacement_map


def create_periodic_mesh(
    mesh, indicator, mapping_function
) -> tuple[dolfinx.mesh.Mesh, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Create a periodic mesh that takes all facets that satisfy the `indicator` function,
    and map the vertices of these facets to the vertices that satisfies the mapping function.

    Note:
        The cell ownership does not change, only additional ghosts are added to a given process

    Note:
        The vertex ownership does not change, only additional ghosts are added to a given process

    Note:
        This is {py:func}`_match_vertices_geometric` followed by {py:func}`_build_periodic_mesh`.
        Only the first half evaluates `indicator` and `mapping_function`; a reader that
        knows the vertex pairs already, such as one for the ``$Periodic`` section of a gmsh
        file, builds a {py:class}`VertexCorrespondence` and calls the second half directly.

    Returns:
        A tuple ``(new_mesh, replaced_vertices, replacement_map)`` where ``new_mesh`` is the new mesh with periodicity,
        ``replaced_vertices`` is a list of vertices of the input mesh that has been replaced (local to process).
        ``replacement_map`` is a map from the old vertices (local to process) to the new vertices (local to process).

        Note:
            This map does not contain additional ghost vertices added to the process that has taken over the facet or given away a facet.

    Example:

        .. code-block:: python
        .. highlight:: python

        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 19)
        def indicator(x):
            return numpy.isclose(x[1], 1)

        def map(x):
            values = x.copy()
            values[1] -= 1
            return values

        periodic_mesh = create_periodic_mesh(mesh, indicator, map)
    """
    return _build_periodic_mesh(
        mesh, _match_vertices_geometric(mesh, indicator, mapping_function)
    )


def create_periodic_mesh_from_gmsh(
    mesh, slave_igi, master_igi, num_nodes_global
) -> tuple[dolfinx.mesh.Mesh, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Make `mesh` periodic from the node pairs of a gmsh ``$Periodic`` block.

    The point of {py:class}`VertexCorrespondence` is that it is the seam between *finding* the
    periodic pairs and *rebuilding* the mesh from them. Everything geometric -- the
    indicator, the mapping function, the tolerance, the point searches -- lives on the
    {py:func}`_match_vertices_geometric` side of it, and {py:func}`_build_periodic_mesh` sees only
    the struct. So a reader that already knows the pairing, as gmsh does, fills the same
    fields and reuses the rebuild unchanged: no `indicator`, no `mapping_function`, and
    therefore no tolerance to tune and no risk of a snap onto the wrong vertex. It also
    handles rotational and reflective periodicity, which the coordinate mapping can only
    express if the caller writes the transform by hand.

    Args:
        mesh: The mesh read from the same gmsh model.
        slave_igi, master_igi: Corresponding node pairs, as 0-based gmsh node tags, i.e.
            values of ``mesh.geometry.input_global_indices``. Held on the reading rank
            only; empty elsewhere. Every master must be a root -- a node that is not
            itself a slave -- so chains through a corner have to be resolved first.
        num_nodes_global: The number of nodes in the gmsh model. Not
            ``mesh.geometry.index_map().size_global``, which is smaller when
            ``create_mesh`` drops nodes no cell references.

    Returns:
        As {py:func}`create_periodic_mesh`.

    Note:
        ``$Periodic`` gives nodes, not facets, so `indicator_facets` has to be derived.
        It follows exactly from `indicator_vertices`: the seam facets are the exterior
        facets all of whose vertices are to be replaced, which is precisely the rule
        `locate_entities_boundary` applies. Take the exterior facets from
        `exterior_facet_indices` and broaden the result, rather than from a local "facet
        with one incident cell" test -- one incident cell locally means the neighbour is
        not ghosted, which is not the same as the facet being exterior, and the broadening
        scatter would carry such a false positive to the owner instead of dropping it.
    """
    raise NotImplementedError(
        "Reading the periodic pairs from gmsh is not implemented yet; build a"
        " `VertexCorrespondence` and call `_build_periodic_mesh` directly."
    )


if __name__ == "__main__":
    # N = 189
    # M = 123
    N = 15
    M = 10

    # mesh = dolfinx.mesh.create_unit_square(
    #     MPI.COMM_WORLD,
    #     N,
    #     M,
    #     ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    #     cell_type=dolfinx.mesh.CellType.quadrilateral,
    # )

    max_facet_to_cell_links = 2
    filename = "mesh.msh"  # python3 create_mesh.py --periodic --res=0.01
    gdim = 2
    comm = MPI.COMM_WORLD
    rank = 0
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet

    if not hasattr(dolfinx.mesh, "create_cell_partitioner"):
        partitioner = dolfinx.graph.partitioner()
    else:
        sig = inspect.signature(dolfinx.mesh.create_cell_partitioner)
        part_kwargs = {}
        if "max_facet_to_cell_links" in sig.parameters:
            part_kwargs["max_facet_to_cell_links"] = max_facet_to_cell_links
            partitioner = dolfinx.mesh.create_cell_partitioner(
                ghost_mode, **part_kwargs
            )

    # NOTE: Add ghost mode once https://github.com/FEniCS/dolfinx/pull/4537 is merged
    sig = inspect.signature(dolfinx.io.gmsh.model_to_mesh)
    if "ghost_mode" in sig.parameters:
        if comm.rank == rank:
            gmsh.initialize()
            gmsh.model.add("Mesh from file")
            gmsh.merge(str(filename))
        mesh_data = dolfinx.io.gmsh.model_to_mesh(
            gmsh.model,
            comm,
            rank,
            gdim=gdim,
            partitioner=partitioner,
            ghost_mode=ghost_mode,
        )
        gmsh.finalize()

    else:
        mesh_data = dolfinx.io.gmsh.read_from_msh(
            filename, MPI.COMM_WORLD, 0, gdim=gdim, partitioner=partitioner
        )

    mesh = mesh_data.mesh
    ct = mesh_data.cell_tags
    ft = mesh_data.facet_tags
    dim = 1
    num_indices_local = mesh.topology.index_map(dim).size_local
    marker = np.arange(num_indices_local, dtype=np.int32)
    tags_old = dolfinx.mesh.meshtags(
        mesh, dim, marker, np.full_like(marker, mesh.comm.rank)
    )

    with dolfinx.io.XDMFFile(mesh.comm, "org_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(tags_old, mesh.geometry)

    L_min = mesh.comm.allreduce(np.min(mesh.geometry.x[:, 0]), op=MPI.MIN)
    L_max = mesh.comm.allreduce(np.max(mesh.geometry.x[:, 0]), op=MPI.MAX)

    def indicator(x):
        return np.isclose(x[0], L_min)

    def mapping(x):
        values = x.copy()
        values[0] += L_max - L_min
        return values

    # mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    # old_num_exterior_facets = mesh.comm.allreduce(len(dolfinx.mesh.exterior_facet_indices(mesh.topology)), op=MPI.SUM)
    # assert old_num_exterior_facets == 2*N + 2*M, "Number of exterior facets is not correct"

    import time

    start = time.perf_counter()
    new_mesh, replaced_vertices, replacement_map = create_periodic_mesh(
        mesh, indicator, mapping
    )
    end = time.perf_counter()
    print(f"Create periodic mesh: {end - start:.3e}")
    if new_mesh.comm.size == 1:
        np.testing.assert_allclose(
            new_mesh.topology.original_cell_index, mesh.topology.original_cell_index
        )

    def num_vertices_per_entity(cell_type: dolfinx.mesh.CellType, dim: int) -> int:
        entity_vertices = dolfinx.cpp.mesh.get_entity_vertices(cell_type, dim)
        num_entity_vertices = entity_vertices.offsets[1:] - entity_vertices.offsets[:-1]

        assert np.unique(num_entity_vertices).size == 1, (
            "Number of vertices per entity is not constant"
        )
        return num_entity_vertices[0]

    # dim = 1
    # def marker_thing(x):
    #     return x[0]<= 1 + 1e-14

    # indices = dolfinx.mesh.locate_entities(mesh,  dim, marker_thing)
    # num_indices_local = mesh.topology.index_map(dim).size_local
    # local_indices = indices[indices < num_indices_local]
    # marker = np.arange(len(local_indices), dtype=np.int32)
    # tags_old = dolfinx.mesh.meshtags(mesh, dim, local_indices, marker)

    # with dolfinx.io.XDMFFile(mesh.comm, "org_mesh.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(mesh)
    #     xdmf.write_meshtags(tags_old, mesh.geometry)

    tags_periodic = transfer_meshtags_to_periodic_mesh(
        mesh, new_mesh, replaced_vertices, ft
    )
    tags_periodic.name = "Periodic mesh tags"
    with dolfinx.io.XDMFFile(new_mesh.comm, "periodic_mesh_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(new_mesh)
        new_mesh.topology.create_connectivity(dim, new_mesh.topology.dim)
        xdmf.write_meshtags(tags_periodic, new_mesh.geometry)

    # tags_periodic = transfer_meshtags_to_periodic_mesh(mesh, new_mesh, replaced_vertices, tags_old)
    # tags_periodic.name = "Periodic mesh tags"
    # with dolfinx.io.XDMFFile(mesh.comm, "periodic_mesh_tags.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(new_mesh)
    #     new_mesh.topology.create_connectivity(dim, new_mesh.topology.dim)
    #     xdmf.write_meshtags(tags_periodic, new_mesh.geometry)

    # new_mesh.topology.create_connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)
    # num_exterior_facets = mesh.comm.allreduce(len(dolfinx.mesh.exterior_facet_indices(new_mesh.topology)), op=MPI.SUM)
    # assert num_exterior_facets == 2*N, "Number of exterior facets is not correct"

    # Debug information
    # new_mesh.topology.create_connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)
    # f_to_c = new_mesh.topology.connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)
    # f_map = new_mesh.topology.index_map(new_mesh.topology.dim-1)
    # f_range = f_map.local_range

    # left_facets = dolfinx.mesh.locate_entities(new_mesh, new_mesh.topology.dim-1, indicator)
    # #owned_left_facets = left_facets[(f_range[0]<= left_facets) & (left_facets < f_range[1])]
    # lfm = dolfinx.mesh.compute_midpoints(new_mesh, new_mesh.topology.dim-1,left_facets)
    # for facet, midpoint in zip(left_facets, lfm):
    #     assert len(f_to_c.links(facet)) == 2, f"{mesh.comm.rank}: Left facet {facet} {midpoint} only connected to {f_to_c.links(facet)} cells"

    # new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)

    # right_facets = dolfinx.mesh.locate_entities(new_mesh, new_mesh.topology.dim-1,lambda x: np.isclose(x[0], 1.0))
    # owned_right_facets = right_facets[(f_range[0]<= right_facets) & (right_facets < f_range[1])]
    # rfm = dolfinx.mesh.compute_midpoints(new_mesh, new_mesh.topology.dim-1, owned_right_facets)
    # for facet, midpoint in zip(owned_right_facets, rfm):
    #     assert len(f_to_c.links(facet)) == 2, f"{mesh.comm.rank}: Left facet {facet} {midpoint} only connected to {f_to_c.links(facet)} cells"

    # new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)

    # with dolfinx.io.XDMFFile(mesh.comm, "periodic_mesh.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(new_mesh)

    x = ufl.SpatialCoordinate(new_mesh)
    u_ex = ufl.sin(2 * np.pi * x[0])
    h = 2 * ufl.Circumradius(new_mesh)
    h_avg = ufl.avg(h)
    gamma = dolfinx.fem.Constant(new_mesh, 100.0)
    alpha = dolfinx.fem.Constant(new_mesh, 100.0)

    V = dolfinx.fem.functionspace(new_mesh, ("DG", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(new_mesh)
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += -ufl.inner(ufl.jump(v, n), ufl.avg(ufl.grad(u))) * ufl.dS
    F += -ufl.inner(ufl.avg(ufl.grad(v)), ufl.jump(u, n)) * ufl.dS
    F += +gamma / h_avg * ufl.inner(ufl.jump(v, n), ufl.jump(u, n)) * ufl.dS

    F += -ufl.inner(n, ufl.grad(u)) * v * ufl.ds

    F += -ufl.inner(n, ufl.grad(v)) * u * ufl.ds + alpha / h * ufl.inner(u, v) * ufl.ds
    F -= (
        -ufl.inner(n, ufl.grad(v)) * u_ex * ufl.ds
        + alpha / h * ufl.inner(u_ex, v) * ufl.ds
    )

    x = ufl.SpatialCoordinate(new_mesh)
    f = 100 ** x[0] * ufl.sin(0.5 * np.pi * x[1])
    F -= ufl.inner(f, v) * ufl.dx
    a, L = ufl.system(F)
    import dolfinx.fem.petsc

    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[],
        petsc_options_prefix="periodic_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
            "ksp_monitor": None,
        },
    )
    uh = problem.solve()

    with dolfinx.io.VTXWriter(new_mesh.comm, "u_periodic.bp", [uh]) as writer:
        writer.write(0.0)
