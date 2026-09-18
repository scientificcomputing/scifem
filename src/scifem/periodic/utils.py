import numpy as np
import numpy.typing as npt
import dataclasses
import dolfinx

from ..mpi_utils import (
    broadcast_marked_entities,
)


@dataclasses.dataclass
class PeriodicNodes:
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


def resolve_to_roots(slave, master):
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
