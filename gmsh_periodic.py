# Read the periodic node correspondence gmsh stores in a model
# SPDX-License-Identifier: MIT

"""Turn a gmsh model's ``$Periodic`` section into a vertex correspondence.

gmsh already knows which nodes a periodic boundary identifies -- it built the mesh that
way -- so the pairing need not be recovered by mapping coordinates and searching for the
nearest vertex. That also makes rotational and reflective periodicity work without the
caller writing the transform by hand.

What gmsh stores is a *relation*, not a function. Pairs are recorded per model entity,
including the dimension-0 point entities, so a corner node appears several times with
different masters: on a doubly periodic square the node at (1,1) is paired with (0,1)
through the right-hand curve and with (1,0) through the top curve. Both routes lead to
(0,0), and resolving to that common root is what this module does.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

import dolfinx

import script


@dataclasses.dataclass
class GmshPeriodicNodes:
    """The node pairs of a gmsh model, resolved to roots.

    Args:
        slave: 0-based gmsh node tags that are to be replaced, ascending and without
            repeats. These are values of ``mesh.geometry.input_global_indices``.
        master: For each entry of `slave`, the node it is identified with. Never itself a
            slave, so no further resolution is needed.
        num_nodes_global: The number of nodes in the gmsh model. Not
            ``mesh.geometry.index_map().size_global``, which is smaller when ``create_mesh``
            drops nodes that no cell references.
    """

    slave: npt.NDArray[np.int64]
    master: npt.NDArray[np.int64]
    num_nodes_global: int


def _resolve_to_roots(slave, master):
    """Follow every pair to a node that is not itself replaced.

    Args:
        slave: 0-based node tags, with repeats and possibly several masters each.
        master: The node paired with each entry of `slave`.

    Returns:
        ``(unique_slave, root)``: each distinct slave once, and the node it ultimately
        resolves to.

    Raises:
        RuntimeError: If the pairs cycle, or if two routes out of one node disagree on
            where it ends up.
    """
    unique_slave, first = np.unique(slave, return_index=True)
    # One master per slave to iterate on. Where a node has several -- a corner -- any one
    # will do, because the agreement check below proves they all lead to the same place.
    next_of = master[first]

    # `position[n]` is where node n sits in `unique_slave`, or -1 if it is already a root.
    lookup = np.full(int(max(unique_slave.max(), master.max())) + 2, -1, dtype=np.int64)
    lookup[unique_slave] = np.arange(len(unique_slave), dtype=np.int64)

    # Pointer doubling: each pass at least halves the remaining chain length, so
    # ``ceil(log2(n)) + 1`` passes suffice unless the pairs cycle.
    root = next_of.copy()
    max_passes = int(np.ceil(np.log2(max(len(unique_slave), 2)))) + 1
    for _ in range(max_passes):
        position = lookup[root]
        moving = position != -1
        if not moving.any():
            break
        root[moving] = root[position[moving]]
    else:
        still = lookup[root] != -1
        raise RuntimeError(
            f"{int(np.count_nonzero(still))} periodic node chains do not end: the"
            " `$Periodic` pairs cycle, so no node is a root. First offending node tag"
            f" (1-based): {int(unique_slave[np.flatnonzero(still)[0]]) + 1}."
        )

    # Every recorded pair has to agree on the root, including the duplicates dropped
    # above. A disagreement means the model itself is inconsistent, not that a route was
    # picked badly.
    def root_of(nodes):
        position = lookup[nodes]
        return np.where(position == -1, nodes, root[np.maximum(position, 0)])

    mismatch = root_of(slave) != root_of(master)
    if mismatch.any():
        i = int(np.flatnonzero(mismatch)[0])
        raise RuntimeError(
            "Inconsistent `$Periodic` section: node tags (1-based)"
            f" {int(slave[i]) + 1} and {int(master[i]) + 1} are recorded as a periodic"
            f" pair but resolve to different roots, {int(root_of(slave[i : i + 1])[0]) + 1}"
            f" and {int(root_of(master[i : i + 1])[0]) + 1}."
        )
    return unique_slave, root


def extract_gmsh_periodic_nodes(
    model, include_high_order: bool = False, tol: float = 1e-8
) -> GmshPeriodicNodes:
    """Collect the ``$Periodic`` node pairs of `model`, resolved to roots.

    Runs where the gmsh model lives, so serially on the reading rank.

    Args:
        model: An initialised ``gmsh.model`` carrying a meshed, periodic geometry.
        include_high_order: Keep the nodes that are not cell vertices. The correspondence
            this feeds is between *vertices*, and on a higher-order mesh
            ``entities_to_geometry(mesh, 0, vertices)`` returns each vertex's corner node,
            so the default is what is wanted. The resolution below treats the extra nodes
            no differently, so the flag costs nothing either way.
        tol: Absolute tolerance for checking each pair against the affine transform gmsh
            recorded with it. Pairs whose entity stored no transform are not checked.

    Returns:
        The pairs, as :class:`GmshPeriodicNodes`.

    Raises:
        RuntimeError: If the pairs cycle, disagree on a root, or contradict the affine
            transform recorded with them.
    """
    slaves, masters, hops = [], [], []
    for dim, tag in model.getEntities():
        master_tag, node_tags, master_node_tags, affine = model.mesh.getPeriodicNodes(
            dim, tag, include_high_order
        )
        # gmsh returns the entity itself as its own master when it is not periodic.
        if master_tag == tag or len(node_tags) == 0:
            continue
        s = np.asarray(node_tags, dtype=np.int64) - 1
        m = np.asarray(master_node_tags, dtype=np.int64) - 1
        slaves.append(s)
        masters.append(m)
        hops.append((s, m, np.asarray(affine, dtype=np.float64)))

    all_node_tags, all_coords, _ = model.mesh.getNodes()
    num_nodes_global = int(np.asarray(all_node_tags, dtype=np.int64).max())

    if not slaves:
        empty = np.zeros(0, dtype=np.int64)
        return GmshPeriodicNodes(empty, empty, num_nodes_global)

    slave = np.concatenate(slaves)
    master = np.concatenate(masters)

    # Coordinates by 0-based tag, for the affine check.
    coords = np.zeros((num_nodes_global, 3), dtype=np.float64)
    coords[np.asarray(all_node_tags, dtype=np.int64) - 1] = np.asarray(
        all_coords, dtype=np.float64
    ).reshape(-1, 3)
    for s, m, affine in hops:
        # gmsh stores a 4x4 row-major matrix, or nothing at all for some entities.
        if affine.size != 16:
            continue
        matrix = affine.reshape(4, 4)
        mapped = coords[m] @ matrix[:3, :3].T + matrix[:3, 3]
        gap = np.linalg.norm(mapped - coords[s], axis=1)
        if (gap > tol).any():
            i = int(np.argmax(gap))
            raise RuntimeError(
                "A `$Periodic` pair does not satisfy the affine transform recorded with"
                f" it: node tags (1-based) {int(m[i]) + 1} and {int(s[i]) + 1} are"
                f" {gap[i]:.3e} apart after the transform, tolerance {tol:.3e}."
            )

    unique_slave, root = _resolve_to_roots(slave, master)
    assert not np.isin(root, unique_slave).any(), "a root is itself replaced"
    return GmshPeriodicNodes(unique_slave, root, num_nodes_global)


def _local_range(comm, num_indices):
    """The block of ``range(num_indices)`` this process is the post office for.

    Uses ``dolfinx.common.local_range`` where it exists, so that the blocks match what the
    rest of DOLFINx means by the same words; the fallback reproduces it.
    """
    if hasattr(dolfinx.common, "local_range"):
        return tuple(dolfinx.common.local_range(comm.rank, int(num_indices), comm.size))
    per_rank, remainder = divmod(int(num_indices), comm.size)
    low = comm.rank * per_rank + min(comm.rank, remainder)
    return low, low + per_rank + (1 if comm.rank < remainder else 0)


def _index_owner(comm, indices, num_indices):
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


def _exchange_to_destinations(comm, dest_ranks, payload):
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


def _vertices_that_can_be_paired(mesh):
    """The local vertices a periodic pair could name, and their input global indices.

    gmsh only pairs boundary entities, so restricting to the vertices of the boundary
    keeps the post office below proportional to the surface rather than the volume.

    Args:
        mesh: The mesh to take the vertices of.

    Returns:
        ``(vertices, igi)``: local vertices on the boundary, owned and ghost, and the
        input global index of each. Collective.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim - 1, 0)
    # `entities_to_geometry` at dimension 0 needs the vertex-to-cell map to find a cell to
    # read each vertex's node from.
    mesh.topology.create_connectivity(0, tdim)
    mesh.topology.create_connectivity(tdim, 0)
    boundary_facets = script.broadcast_marked_entities(
        mesh, tdim - 1, dolfinx.mesh.exterior_facet_indices(mesh.topology)
    )
    vertices = dolfinx.mesh.compute_incident_entities(
        mesh.topology, boundary_facets, tdim - 1, 0
    ).astype(np.int32)
    nodes = dolfinx.mesh.entities_to_geometry(mesh, 0, vertices).reshape(-1)
    return vertices, mesh.geometry.input_global_indices[nodes].astype(np.int64)


def _seam_facets_from_vertices(mesh, indicator_vertices):
    """The exterior facets all of whose vertices are in `indicator_vertices`.

    This is exactly what ``locate_entities_boundary`` at ``tdim - 1`` returns for the
    marker that produced `indicator_vertices`: it keeps a facet when every vertex of it is
    marked, over the owned exterior facets. Deriving it means the caller need not have a
    marker function at all.

    Exteriority comes from ``exterior_facet_indices``, which is owned-only and then
    broadened. A local test for a facet with one incident cell would be wrong: one
    incident cell locally means the neighbouring cell is not ghosted, which is not the
    same as the facet being exterior, and the broadening would carry the mistake to the
    owner rather than drop it.

    Args:
        mesh: The mesh the vertices are local to.
        indicator_vertices: Local vertices that are to be replaced, owned and ghost.

    Returns:
        Local facets on the seam, owned and ghost. Collective.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim - 1, 0)
    f_to_v = mesh.topology.connectivity(tdim - 1, 0)

    vertex_map = mesh.topology.index_map(0)
    is_indicator = np.zeros(
        vertex_map.size_local + vertex_map.num_ghosts, dtype=np.bool_
    )
    is_indicator[indicator_vertices] = True

    exterior = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    num_facet_vertices = int(f_to_v.offsets[1] - f_to_v.offsets[0])
    facet_vertices = f_to_v.array[
        (
            f_to_v.offsets[exterior][:, None]
            + np.arange(num_facet_vertices, dtype=np.int32)
        ).reshape(-1)
    ].reshape(len(exterior), num_facet_vertices)
    on_seam = exterior[is_indicator[facet_vertices].all(axis=1)]
    return script.broadcast_marked_entities(mesh, tdim - 1, on_seam)


def periodic_correspondence_from_nodes(
    mesh, pairs: GmshPeriodicNodes, root: int = 0
) -> script.VertexCorrespondence:
    """Turn gmsh node pairs held on one rank into a distributed vertex correspondence.

    The pairs arrive as input global node indices on the reading rank, while the vertices
    they name are spread over every rank, and neither side knows where the other is. A
    post office resolves that: input global index ``i`` is looked after by a fixed rank,
    :func:`_index_owner`, which every process can compute without asking anyone.

    1. every process registers the boundary vertices it holds with the post offices for
       their indices, saying whether it owns each one;
    2. the reader sends each pair to the post office of its *master* index, which knows
       who owns that vertex. It tells that owner which pair it answers, and forwards the
       pair to the post office of the *slave* index, which passes it to every process
       holding a copy;
    3. those processes then ask the master's owner directly, which is what tells it who
       needs the cells at that vertex.

    Only the boundary vertices are registered, since gmsh pairs nothing else, so the post
    office stays proportional to the surface. Nothing is gathered: no process holds more
    than its own block of indices, except the reader, which holds the file it read.

    The rank named for a master is its *vertex* owner, which is unique -- keeping the join
    single-valued -- and always owns a cell incident to the vertex, which is what
    :attr:`script.VertexCorrespondence.src_owner` requires.

    Collective.

    Args:
        mesh: The mesh built from the same gmsh model, so that
            ``mesh.geometry.input_global_indices`` is the node numbering `pairs` uses.
        pairs: The node pairs, meaningful on `root` only.
        root: The rank holding `pairs`.

    Returns:
        The correspondence :func:`script._build_periodic_mesh` consumes.

    Raises:
        RuntimeError: If a pair names a node that is not a vertex of the mesh.
    """
    comm = mesh.comm
    num_owned_vertices = mesh.topology.index_map(0).size_local
    num_nodes_global = comm.bcast(
        pairs.num_nodes_global if comm.rank == root else None, root=root
    )

    # (1) Register the boundary vertices with the post offices for their indices.
    local_vertices, local_igi = _vertices_that_can_be_paired(mesh)
    owns = (local_vertices < num_owned_vertices).astype(np.int64)
    registrar, registered = _exchange_to_destinations(
        comm,
        _index_owner(comm, local_igi, num_nodes_global),
        np.stack([local_igi, owns], axis=1),
    )
    held_igi, held_owns = registered[:, 0], registered[:, 1].astype(bool)

    # Index the register by its own block, so a lookup is one array read.
    low, high = _local_range(comm, num_nodes_global)
    owner_of = np.full(max(high - low, 0), -1, dtype=np.int64)
    owner_of[held_igi[held_owns] - low] = registrar[held_owns]

    # (2) The reader hands each pair to the post office for its master index.
    if comm.rank == root:
        to_master_office = np.stack(
            [
                np.asarray(pairs.master, dtype=np.int64),
                np.asarray(pairs.slave, dtype=np.int64),
                np.arange(len(pairs.slave), dtype=np.int64),
            ],
            axis=1,
        )
    else:
        to_master_office = np.zeros((0, 3), dtype=np.int64)
    _, at_master_office = _exchange_to_destinations(
        comm,
        _index_owner(comm, to_master_office[:, 0], num_nodes_global),
        to_master_office,
    )
    master_igi, slave_igi, pair_id = at_master_office.T

    unknown = owner_of[master_igi - low] == -1 if len(master_igi) else np.zeros(0, bool)
    # Collective: only the post offices see this, so it is reduced before anyone raises.
    num_unknown = comm.allreduce(int(np.count_nonzero(unknown)), op=MPI.SUM)
    if num_unknown:
        raise RuntimeError(
            f"{num_unknown} periodic pairs name a master node that is not a boundary"
            " vertex of the distributed mesh. The pairs and the mesh have to come from"
            " the same gmsh model, and the master nodes have to be cell vertices."
        )
    master_owner = owner_of[master_igi - low]

    # Tell each master's owner which pair its vertex answers.
    _, assignment = _exchange_to_destinations(
        comm, master_owner.astype(np.int32), np.stack([pair_id, master_igi], axis=1)
    )
    answers_pair, answers_igi = assignment[:, 0], assignment[:, 1]

    # Forward the pair to the post office for its slave index, which knows the holders.
    _, at_slave_office = _exchange_to_destinations(
        comm,
        _index_owner(comm, slave_igi, num_nodes_global),
        np.stack([slave_igi, pair_id, master_owner], axis=1),
    )
    wanted_igi, wanted_pair, wanted_owner = at_slave_office.T

    # Fan out to every process holding a copy of the slave vertex, ghosts included: each
    # of them carries the vertex in `indicator_vertices` and needs its own `src_owner`.
    holder_order = np.argsort(held_igi, kind="stable")
    holder_igi = held_igi[holder_order]
    holder_rank = registrar[holder_order]
    first = np.searchsorted(holder_igi, wanted_igi, side="left")
    last = np.searchsorted(holder_igi, wanted_igi, side="right")
    repeats = last - first
    fan = np.repeat(np.arange(len(wanted_igi)), repeats)
    within = np.arange(len(fan)) - np.repeat(np.cumsum(repeats) - repeats, repeats)
    _, delivered = _exchange_to_destinations(
        comm,
        holder_rank[first[fan] + within],
        np.stack([wanted_igi[fan], wanted_pair[fan], wanted_owner[fan]], axis=1),
    )

    # Back in local numbering. A vertex is delivered once per route that reaches it, and
    # the correspondence wants it once.
    igi_order = np.argsort(local_igi, kind="stable")
    position = np.searchsorted(local_igi[igi_order], delivered[:, 0])
    my_vertex = local_vertices[igi_order][position]
    indicator_vertices, keep = np.unique(my_vertex, return_index=True)
    indicator_vertices = indicator_vertices.astype(np.int32)
    src_owner = delivered[keep, 2].astype(np.int32)
    my_pair = delivered[keep, 1]

    # (3) Ask the master's owner directly. The request is what tells it who needs the
    # cells at that vertex, which is what `dest_owner` records.
    dest_owner, requested = _exchange_to_destinations(
        comm, src_owner, my_pair.reshape(-1, 1)
    )

    # Resolve each request to the local vertex answering it, via step (2)'s assignment.
    answer_order = np.argsort(answers_pair, kind="stable")
    answer_pair = answers_pair[answer_order]
    answer_igi = answers_igi[answer_order]
    position = np.searchsorted(answer_pair, requested[:, 0])
    # A process can own no master vertex at all, and then receives no request either.
    # Both arrays are empty and there is nothing to check, so the emptiness of the
    # assignments must not by itself count as a failure.
    found = (
        np.zeros(len(requested), dtype=np.bool_)
        if len(answer_pair) == 0
        else answer_pair[np.minimum(position, len(answer_pair) - 1)] == requested[:, 0]
    )
    assert found.all(), "a request reached a rank that was not assigned that pair"
    position = np.searchsorted(local_igi[igi_order], answer_igi[position])
    partner_vertex = local_vertices[igi_order][position].astype(np.int32)

    # The correspondence requires `dest_owner` ascending; `_exchange_to_destinations`
    # sorts by source rank for exactly this.
    assert np.all(dest_owner[:-1] <= dest_owner[1:]), (
        "destination owners are not sorted"
    )

    return script.VertexCorrespondence(
        indicator_vertices=indicator_vertices,
        indicator_facets=_seam_facets_from_vertices(mesh, indicator_vertices),
        src_owner=src_owner,
        dest_owner=dest_owner.astype(np.int32),
        partner_vertex=partner_vertex,
    )


def read_periodic_mesh_from_msh(
    filename, comm, rank: int = 0, gdim: int = 3, partitioner=None, **kwargs
):
    """Read a ``.msh`` file and make the mesh periodic from its ``$Periodic`` section.

    Owns the gmsh session, because the pairs have to be read out of the model *before* it
    is finalized and the usual readers finalize it on the way out.

    Collective.

    Args:
        filename: The ``.msh`` file. Read on `rank` only.
        comm: The communicator to distribute the mesh over.
        rank: The rank that reads the file.
        gdim: Geometric dimension of the mesh.
        partitioner: Cell partitioner, passed through to ``model_to_mesh``.
        kwargs: Further arguments for ``model_to_mesh``, such as ``ghost_mode`` where the
            installed DOLFINx takes it there.

    Returns:
        ``(periodic_mesh, replaced_vertices, replacement_map)``, as
        :func:`script.create_periodic_mesh`.
    """
    import gmsh

    started_here = False
    try:
        if comm.rank == rank:
            if not gmsh.isInitialized():
                gmsh.initialize()
                started_here = True
            gmsh.model.add("periodic mesh from file")
            gmsh.merge(str(filename))
            pairs = extract_gmsh_periodic_nodes(gmsh.model)
        else:
            empty = np.zeros(0, dtype=np.int64)
            pairs = GmshPeriodicNodes(empty, empty, 0)

        mesh_data = dolfinx.io.gmsh.model_to_mesh(
            gmsh.model, comm, rank, gdim=gdim, partitioner=partitioner, **kwargs
        )
    finally:
        if started_here and gmsh.isInitialized():
            gmsh.finalize()

    mesh = getattr(mesh_data, "mesh", mesh_data)
    return script.create_periodic_mesh_from_gmsh(
        mesh, pairs.slave, pairs.master, pairs.num_nodes_global, root=rank
    )
