from mpi4py import MPI
import numpy as np

from ..mpi_utils import (
    exchange_to_destinations,
    index_owner,
    local_range,
    broadcast_marked_entities,
)
from .utils import (
    PeriodicNodes,
    VertexCorrespondence,
)
import dolfinx


def periodic_correspondence_from_nodes(
    mesh, pairs: PeriodicNodes, root: int = 0
) -> VertexCorrespondence:
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
    # The largest tag rides along with the count so the check below costs no extra
    # collective, and so every process can raise the same message.
    num_nodes_global, largest_tag = comm.bcast(
        (
            pairs.num_nodes_global,
            int(max(pairs.slave.max(), pairs.master.max())) if len(pairs.slave) else -1,
        )
        if comm.rank == root
        else None,
        root=root,
    )
    # `num_nodes_global` keys every post office below, so one that is too small does not
    # fail -- it misroutes, and the pairs quietly go to the wrong ranks. It is the one
    # field of the correspondence that cannot be derived from the mesh, and the one a
    # caller that took the default would leave at zero.
    if largest_tag >= num_nodes_global:
        raise RuntimeError(
            f"`num_nodes_global` is {num_nodes_global}, but the pairs name node"
            f" {largest_tag}. It has to be the gmsh model's node count, from"
            " `getNodes()` -- not `mesh.geometry.index_map().size_global`, which is"
            " smaller whenever `create_mesh` drops a node no cell references."
        )

    # (1) Register the boundary vertices with the post offices for their indices.
    local_vertices, local_igi = _vertices_that_can_be_paired(mesh)
    owns = (local_vertices < num_owned_vertices).astype(np.int64)
    registrar, registered = exchange_to_destinations(
        comm,
        index_owner(comm, local_igi, num_nodes_global),
        np.stack([local_igi, owns], axis=1),
    )
    held_igi, held_owns = registered[:, 0], registered[:, 1].astype(bool)

    # Index the register by its own block, so a lookup is one array read.
    low, high = local_range(comm, num_nodes_global)
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
    _, at_master_office = exchange_to_destinations(
        comm,
        index_owner(comm, to_master_office[:, 0], num_nodes_global),
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
    _, assignment = exchange_to_destinations(
        comm, master_owner.astype(np.int32), np.stack([pair_id, master_igi], axis=1)
    )
    answers_pair, answers_igi = assignment[:, 0], assignment[:, 1]

    # Forward the pair to the post office for its slave index, which knows the holders.
    _, at_slave_office = exchange_to_destinations(
        comm,
        index_owner(comm, slave_igi, num_nodes_global),
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
    _, delivered = exchange_to_destinations(
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
    dest_owner, requested = exchange_to_destinations(comm, src_owner, my_pair.reshape(-1, 1))

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
    assert np.all(dest_owner[:-1] <= dest_owner[1:]), "destination owners are not sorted"

    return VertexCorrespondence(
        indicator_vertices=indicator_vertices,
        indicator_facets=_seam_facets_from_vertices(mesh, indicator_vertices),
        src_owner=src_owner,
        dest_owner=dest_owner.astype(np.int32),
        partner_vertex=partner_vertex,
    )


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
    boundary_facets = broadcast_marked_entities(
        mesh, tdim - 1, dolfinx.mesh.exterior_facet_indices(mesh.topology)
    )
    # A process can hold a copy of a boundary vertex without holding any boundary facet at
    # it -- it may ghost a cell at the vertex whose own facets there are all interior --
    # and it still has to register, or the fan-out will not reach it and it will disagree
    # with the others about which vertices are replaced. Broadcasting the set closes that:
    # the processes that do see a boundary facet mark the vertex, and the reduce-then-
    # scatter carries the mark to every holder.
    vertices = broadcast_marked_entities(
        mesh,
        0,
        dolfinx.mesh.compute_incident_entities(mesh.topology, boundary_facets, tdim - 1, 0).astype(
            np.int32
        ),
    )
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
    is_indicator = np.zeros(vertex_map.size_local + vertex_map.num_ghosts, dtype=np.bool_)
    is_indicator[indicator_vertices] = True

    exterior = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    num_facet_vertices = int(f_to_v.offsets[1] - f_to_v.offsets[0])
    facet_vertices = f_to_v.array[
        (f_to_v.offsets[exterior][:, None] + np.arange(num_facet_vertices, dtype=np.int32)).reshape(
            -1
        )
    ].reshape(len(exterior), num_facet_vertices)
    on_seam = exterior[is_indicator[facet_vertices].all(axis=1)]
    return broadcast_marked_entities(mesh, tdim - 1, on_seam)
