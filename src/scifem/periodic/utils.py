import numpy as np
import numpy.typing as npt
import dataclasses


__all__ = [
    "PeriodicNodes",
    "VertexCorrespondence",
    "resolve_to_roots",
]


@dataclasses.dataclass
class PeriodicNodes:
    """Node pairs to identify, resolved to roots.

    The pairs are given in the mesh's input global numbering, so they say nothing about how
    the mesh is distributed and can come from anywhere that knows it --
    :py:func:`scifem.periodic.extract_gmsh_periodic_nodes` reads them out of a
    ``$Periodic`` section, but nothing here depends on that.

    Only the process that has the pairs holds them; every other one passes an empty set,
    which is what the defaults are for -- ``PeriodicNodes()`` says "nothing here" without
    the caller having to build two empty arrays to say it.

    Args:
        replaced: 0-based node indices that are to be replaced, ascending and without
            repeats. These are values of
            :py:attr:`input_global_indices<dolfinx.mesh.Geometry.input_global_indices>`.
        partner: For each entry of `replaced`, the node it is identified with. Never itself
            replaced, so no further resolution is needed.
        num_nodes_global: The size of the input global numbering, i.e. one past its largest
            index. Not ``mesh.geometry.index_map().size_global``, which is smaller whenever
            the mesh was built from a node set with entries no cell references. Taken from
            `root` and broadcast, so the default stands on every other process; on `root` it
            has to be set, and :py:func:`periodic_correspondence_from_nodes` checks that it
            was.
    """

    replaced: npt.NDArray[np.int64] = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    partner: npt.NDArray[np.int64] = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    num_nodes_global: int = 0


@dataclasses.dataclass
class VertexCorrespondence:
    """Which vertices of ``mesh`` are identified with which, and which ranks hold each end.

    This is what :py:mod:`scifem.periodic` rebuilds from: it consumes nothing else and
    never evaluates a coordinate, which is what lets the pairs be found either
    geometrically or topologically.

    Stores the data of :py:class:`dolfinx.geometry.PointOwnershipData` for the
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


def resolve_to_roots(replaced, partner):
    """Follow every pair to a node that is not itself replaced.

    Args:
        replaced: 0-based node tags, with repeats and possibly several partners each.
        partner: The node paired with each entry of `replaced`.

    Returns:
        ``(unique_replaced, root)``: each distinct replaced once, and the node it ultimately
        resolves to.

    Raises:
        RuntimeError: If the pairs cycle, or if two routes out of one node disagree on
            where it ends up.
    """
    unique_replaced, first = np.unique(replaced, return_index=True)
    # One partner per node to iterate on. Where a node has several -- a corner -- any one
    # will do, because the agreement check below proves they all lead to the same place.
    next_of = partner[first]

    # `position[n]` is where node n sits in `unique_replaced`, or -1 if it is already a root.
    lookup = np.full(int(max(unique_replaced.max(), partner.max())) + 2, -1, dtype=np.int64)
    lookup[unique_replaced] = np.arange(len(unique_replaced), dtype=np.int64)

    # Pointer doubling: each pass at least halves the remaining chain length, so
    # ``ceil(log2(n)) + 1`` passes suffice unless the pairs cycle.
    root = next_of.copy()
    max_passes = int(np.ceil(np.log2(max(len(unique_replaced), 2)))) + 1
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
            f" (1-based): {int(unique_replaced[np.flatnonzero(still)[0]]) + 1}."
        )

    # Every recorded pair has to agree on the root, including the duplicates dropped
    # above. A disagreement means the model itself is inconsistent, not that a route was
    # picked badly.
    def root_of(nodes):
        position = lookup[nodes]
        return np.where(position == -1, nodes, root[np.maximum(position, 0)])

    mismatch = root_of(replaced) != root_of(partner)
    if mismatch.any():
        i = int(np.flatnonzero(mismatch)[0])
        raise RuntimeError(
            "Inconsistent `$Periodic` section: node tags (1-based)"
            f" {int(replaced[i]) + 1} and {int(partner[i]) + 1} are recorded as a periodic"
            f" pair but resolve to different roots, {int(root_of(replaced[i : i + 1])[0]) + 1}"
            f" and {int(root_of(partner[i : i + 1])[0]) + 1}."
        )
    return unique_replaced, root
