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


def _exchange_to_destinations(comm, dest_ranks, payload):
    """Send one payload entry to each named rank, and receive whatever arrives.

    Only the outgoing edges are known here -- a process cannot tell in advance who will
    ask it for a vertex -- so the neighbourhood is built with ``Create_dist_graph``, which
    derives the incoming edges, rather than ``Create_dist_graph_adjacent``.

    Args:
        comm: The communicator to exchange over. Collective.
        dest_ranks: Destination rank of each entry of `payload`. Need not be sorted.
        payload: One ``int64`` value per entry.

    Returns:
        ``(sources, received)``: the rank each received entry came from, ascending, and
        the values, grouped by source in the same order.
    """
    order = np.argsort(dest_ranks, kind="stable")
    dests, counts = np.unique(dest_ranks, return_counts=True)
    send_buffer = np.ascontiguousarray(payload[order], dtype=np.int64)

    graph = comm.Create_dist_graph(
        [comm.rank], [len(dests)], dests.astype(np.int32).tolist(), MPI.UNWEIGHTED
    )
    try:
        in_ranks, _, _ = graph.Get_dist_neighbors()
        in_ranks = np.asarray(in_ranks, dtype=np.int32)

        # Uniform neighbourhood collective: the count is passed explicitly and is the same
        # on every process, as MPI-4.1 9.6.2 requires. Letting mpi4py infer it from the
        # buffer size makes it rank-local, which is an erroneous call.
        recv_counts = np.zeros(len(in_ranks), dtype=np.int32)
        graph.Neighbor_alltoall(
            [counts.astype(np.int32), 1, MPI.INT32_T],
            [recv_counts, 1, MPI.INT32_T],
        )

        received = np.zeros(int(recv_counts.sum()), dtype=np.int64)
        graph.Neighbor_alltoallv(
            [send_buffer, counts.astype(np.int32), MPI.INT64_T],
            [received, recv_counts, MPI.INT64_T],
        )
    finally:
        graph.Free()

    # `Get_dist_neighbors` lists the neighbours in the order MPI chose, not ascending, and
    # the receive buffer follows that order. Sort so the caller can rely on the grouping.
    sources = np.repeat(in_ranks, recv_counts).astype(np.int32)
    order = np.argsort(sources, kind="stable")
    return sources[order], received[order]


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

    The pairs arrive as global input node indices on the reading rank, while the vertices
    they name are spread over every rank. Three rounds close that gap, the first two of
    them reusing :func:`dolfinx.io.distribute_entity_data`, which already delivers data
    keyed by input global index to the ranks that hold the entity:

    1. send each pair's index to the ranks holding its *master* vertex, so the owner of
       that vertex can say so;
    2. gather those answers on `root` and send them back out keyed by the *slave* vertex,
       so every rank holding a replaced vertex learns which rank holds its partner;
    3. ask that rank directly, which is what tells it who to send cells to.

    The rank naming the master's owner is deliberately its *vertex* owner. That rank always
    owns a cell incident to the vertex, so it satisfies the contract on
    :attr:`script.VertexCorrespondence.src_owner`, and unlike a cell owner it is unique,
    which keeps the join single-valued.

    Collective.

    Args:
        mesh: The mesh built from the same gmsh model, so that
            ``mesh.geometry.input_global_indices`` refers to the same node numbering as
            `pairs`.
        pairs: The node pairs, meaningful on `root` only.
        root: The rank holding `pairs`.

    Returns:
        The correspondence :func:`script._build_periodic_mesh` consumes.
    """
    comm = mesh.comm
    vertex_map = mesh.topology.index_map(0)
    num_owned_vertices = vertex_map.size_local

    if comm.rank == root:
        slave_igi = np.asarray(pairs.slave, dtype=np.int64)
        master_igi = np.asarray(pairs.master, dtype=np.int64)
    else:
        slave_igi = np.zeros(0, dtype=np.int64)
        master_igi = np.zeros(0, dtype=np.int64)
    num_pairs = comm.bcast(len(slave_igi), root=root)
    pair_index = np.arange(len(slave_igi), dtype=np.int64)

    # (1) Which pairs' master vertices does this rank hold, and which does it own?
    master_vertex, master_pair = dolfinx.io.distribute_entity_data(
        mesh, 0, master_igi.reshape(-1, 1), pair_index
    )
    master_vertex = np.asarray(master_vertex, dtype=np.int64).reshape(-1)
    master_pair = np.asarray(master_pair, dtype=np.int64).reshape(-1)
    owns_master = master_vertex < num_owned_vertices

    # (2) Report those to `root`, which turns them into a value per pair and sends it back
    # out keyed by the slave vertex. Exactly one rank owns each master, so the array is
    # filled exactly once; -1 would mean a master that no rank holds.
    announced = comm.gather(master_pair[owns_master], root=root)
    announced_by = comm.gather(
        np.full(int(np.count_nonzero(owns_master)), comm.rank, dtype=np.int64),
        root=root,
    )
    if comm.rank == root:
        master_owner = np.full(num_pairs, -1, dtype=np.int64)
        for which, by in zip(announced, announced_by):
            master_owner[which] = by
        missing = int(np.count_nonzero(master_owner == -1))
    else:
        master_owner = np.zeros(0, dtype=np.int64)
        missing = 0
    # Collective: `root` alone knows, but a raise there alone would leave the others
    # blocked in the exchanges below.
    missing = comm.bcast(missing, root=root)
    if missing:
        raise RuntimeError(
            f"{missing} periodic pairs name a master node that is not a vertex of the"
            " distributed mesh. The pairs and the mesh have to come from the same gmsh"
            " model, and the master nodes have to be cell vertices."
        )

    indicator_vertex, partner_rank = dolfinx.io.distribute_entity_data(
        mesh, 0, slave_igi.reshape(-1, 1), master_owner
    )
    indicator_vertex = np.asarray(indicator_vertex, dtype=np.int64).reshape(-1)
    partner_rank = np.asarray(partner_rank, dtype=np.int64).reshape(-1)
    _, which_pair = dolfinx.io.distribute_entity_data(
        mesh, 0, slave_igi.reshape(-1, 1), pair_index
    )
    which_pair = np.asarray(which_pair, dtype=np.int64).reshape(-1)

    # `distribute_entity_data` returns a vertex once per holder, so the same local vertex
    # can come back more than once; the correspondence wants it once.
    indicator_vertices, keep = np.unique(indicator_vertex, return_index=True)
    indicator_vertices = indicator_vertices.astype(np.int32)
    src_owner = partner_rank[keep].astype(np.int32)

    # (3) Ask the rank holding the partner. It learns from the request who needs the cells
    # at that vertex, which is what `dest_owner` records.
    dest_owner, requested_pair = _exchange_to_destinations(
        comm, src_owner, which_pair[keep]
    )

    # Resolve each requested pair to the local vertex that answers it.
    owned_pair = master_pair[owns_master]
    owned_vertex = master_vertex[owns_master]
    order = np.argsort(owned_pair, kind="stable")
    position = np.searchsorted(owned_pair[order], requested_pair)
    assert (position < len(owned_pair)).all() and (
        owned_pair[order][np.minimum(position, len(owned_pair) - 1)] == requested_pair
    ).all(), "a request reached a rank that does not own that pair's master vertex"
    partner_vertex = owned_vertex[order][position].astype(np.int32)

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
    return script._build_periodic_mesh(
        mesh, periodic_correspondence_from_nodes(mesh, pairs, root=rank)
    )
