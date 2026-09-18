# Regression tests for create_periodic_mesh on a constructed partition
# SPDX-License-Identifier: MIT

"""The ghost layer `create_periodic_mesh` builds, pinned with a partition of our own.

A doubly periodic square is a torus: it has no boundary, so in the rebuilt mesh *every
owned facet must have exactly two incident cells*. Two ghosting bugs used to leave one
of the seam facets with a single cell, which shows up at assembly time as
``Cannot compute interior facet integral over interprocess facet``:

* Phase 1 shipped only the cells behind the boundary facets a process *owned*
  (`dolfinx.mesh.exterior_facet_indices` is owned-only). A process that took over a
  replacement vertex while merely *ghosting* one of the boundary facets touching it
  never shipped the cell behind that facet.
* Phase 3 shipped the cell to the single owner `dolfinx.geometry.determine_point_ownership`
  returned for the mapped point. That point sits exactly on a vertex, which several cells
  share, so the choice among them is arbitrary and the other processes owning cells
  incident to that replacement vertex were left short.

Neither is really about the number of processes: both need particular ranks to meet at a
single seam vertex. Under the default partitioner that happened to occur at 5 ranks on a
4x4 mesh and nowhere near it, which is what made the bugs look rank-count specific. Here
the partition is built rather than stumbled upon -- `seam_split_partitioner` below puts a
rank boundary through the seam vertices and keeps the halos thin -- and both bugs then
appear at 2, 3 and 4 ranks (verified against the pre-fix code).

Run serially, or under MPI::

    python3 -m pytest test_partition.py
    mpirun -n 3 python3 -m pytest test_partition.py
"""

from mpi4py import MPI

import basix.ufl
import numpy as np
import pytest
import ufl

import dolfinx

from scifem.periodic.mesh import create_periodic_mesh

# A hand-written geometric partitioner needs wrapping before `create_mesh` will take it,
# and the wrappers only exist on DOLFINx main (0.12.x); 0.11 has no way to pass cell
# positions to a partitioner at all, so there the tests below have nothing to build on.
_HAS_GEOMETRIC_PARTITIONER = hasattr(dolfinx.mesh, "create_geometric_cell_partitioner")
# The partitioning decision here is purely geometric, but a geometric partitioner never
# ghosts (`create_geometric_cell_partitioner` forces `GhostMode.none`), and both the
# periodic merge and the `dS` assembly below need a ghost layer. The hybrid wrapper is the
# one that also hands the partitioner the dual graph, which is all that is needed to work
# out ghost destinations. It lands in the same DOLFINx versions.
_HAS_HYBRID_PARTITIONER = hasattr(dolfinx.mesh, "create_hybrid_cell_partitioner")

pytestmark = pytest.mark.skipif(
    not (_HAS_GEOMETRIC_PARTITIONER and _HAS_HYBRID_PARTITIONER),
    reason=(
        "custom geometric partitioning needs dolfinx.mesh.create_geometric_cell_partitioner"
        " and create_hybrid_cell_partitioner, which are only on DOLFINx main"
    ),
)

# Cells per direction. 12 is the size that forces both bugs at 2, 3 and 4 ranks; 6 and 8
# force them at some of those counts and are kept as extra cover, since which rank ends up
# short depends on details of the numbering that a DOLFINx change could shift. 3 is the
# smallest usable size in principle -- a 2x2 doubly periodic mesh is non-manifold, each cell
# meeting its neighbour on both sides, so every facet gets four cells -- but the halos of a
# mesh that small cover the whole domain and hide any missing cell.
GRID_SIZES = (6, 8, 12)

# Band width numerator/denominator: bands are 3/2 cells wide, see `seam_split_partitioner`
_BAND_NUM, _BAND_DEN = 2, 3


# --------------------------------------------------------------------------- #
# A partition that puts a rank boundary through the seam
# --------------------------------------------------------------------------- #


def num_bands(n):
    """Number of distinct bands `seam_split_partitioner` cuts an ``n x n`` grid into."""
    return (_BAND_NUM * (2 * n - 2)) // _BAND_DEN + 1


def seam_split_partitioner(n):
    """Geometric partitioner for the ``n x n`` unit square: anti-diagonal bands, dealt out.

    Cell ``(i, j)`` goes to rank ``floor(2 * (i + j) / 3) % nparts``: the anti-diagonals
    ``i + j = const`` are grouped into bands 3/2 cells wide -- alternately one and two
    diagonals -- and the bands are dealt round-robin to the ranks. Three properties matter,
    and a partitioner that cuts the square into compact pieces has none of them:

    * A band boundary runs through the seam vertices, so the cells meeting at a vertex on
      ``x = 1`` or ``y = 1`` -- a *replacement* vertex, the one a seam vertex is merged into
      -- are owned by more than one rank. That is what phase 3 got wrong by shipping the
      cell to the single cell owner `determine_point_ownership` picked out of several.
    * The bands run diagonally and the width alternates, so they are not commensurate with
      the periodic identification: the rank owning a cell at the ``x = 1`` seam is not the
      rank owning the cell it is glued to at ``x = 0``, and a boundary facet on the seam is
      regularly owned by one rank and merely ghosted by the rank next to it. That is what
      phase 1 used to skip over, `exterior_facet_indices` being owned-only.
    * The bands are still compact, so halos stay thin. On a partition scattered cell by cell
      every process ghosts everything and a cell that was never shipped is present anyway,
      which hides the bug rather than exposing it.

    `n` is fixed by the caller because the rule is stated in cells; the returned function has
    the signature and role of `slab_partitioner` in DOLFINx's `demo_partition.py` -- centroids
    in, one destination rank per centroid out.

    Args:
        n: Cells per direction of the grid being partitioned.

    Returns:
        A geometric partitioning function ``(comm, nparts, x, node_weights) -> ranks``.
    """

    def partitioner(comm, nparts, x, node_weights=None):
        i = np.floor(x[:, 0] * n).astype(np.int64)  # centroid of cell (i, j) is
        j = np.floor(x[:, 1] * n).astype(np.int64)  # ((i + 1/2)/n, (j + 1/2)/n)
        return ((_BAND_NUM * (i + j) // _BAND_DEN) % nparts).astype(np.int32)

    return partitioner


def _centroids(cells_global, n):
    """Centroids of the cells with the given global indices, from the numbering below."""
    j, i = np.divmod(np.asarray(cells_global, dtype=np.int64), n)
    return np.stack([i + 0.5, j + 0.5], axis=1) / n


def _with_ghost_layer(part, n):
    """Wrap a geometric partitioner so it also names ghost destinations.

    A geometric partitioner is not given the dual graph and so cannot ghost. The hybrid
    interface passes both, and the ghost destinations of a cell are just the owners of its
    neighbours in the dual graph. Those neighbours arrive as *global* cell indices, and the
    grid is built here, so their centroids follow from the numbering and the same purely
    geometric rule decides their owners -- no communication needed.
    """

    def hybrid(comm, nparts, graph, x, node_weights, edge_weights, ghosting):
        owner = part(comm, nparts, x, node_weights)
        neighbour_owner = part(comm, nparts, _centroids(graph.array, n), None)
        dests, offsets = [], [0]
        for cell in range(graph.num_nodes):
            d = [int(owner[cell])]
            if ghosting:
                begin, end = graph.offsets[cell], graph.offsets[cell + 1]
                d += sorted({int(r) for r in neighbour_owner[begin:end]} - {d[0]})
            dests += d
            offsets.append(len(dests))
        return dolfinx.graph.adjacencylist(
            np.asarray(dests, dtype=np.int32), np.asarray(offsets, dtype=np.int32)
        )

    return hybrid


def unit_square_block(comm, n):
    """This rank's block of the cells and points of an ``n x n`` quadrilateral unit square.

    Cell ``i + n * j`` is the cell of the grid at column ``i``, row ``j``, and point
    ``i + (n + 1) * j`` is the grid point at ``(i/n, j/n)``. `seam_split_partitioner` and
    `_centroids` both depend on that numbering. Cells and points are dealt out over the
    ranks in contiguous blocks, independently of each other, as `create_mesh` expects.
    """
    first, last = (
        (n * n * comm.rank) // comm.size,
        (n * n * (comm.rank + 1)) // comm.size,
    )
    c = np.arange(first, last, dtype=np.int64)
    cj, ci = np.divmod(c, n)

    def vertex(di, dj):
        return (cj + dj) * (n + 1) + ci + di

    cells = np.stack([vertex(0, 0), vertex(1, 0), vertex(0, 1), vertex(1, 1)], axis=1).astype(
        np.int64
    )

    num_points = (n + 1) ** 2
    first = (num_points * comm.rank) // comm.size
    last = (num_points * (comm.rank + 1)) // comm.size
    p = np.arange(first, last, dtype=np.int64)
    pj, pi = np.divmod(p, n + 1)
    x = np.stack([pi, pj], axis=1).astype(np.float64) / n
    return np.ascontiguousarray(cells), np.ascontiguousarray(x)


def split_seam_mesh(comm, n):
    """An ``n x n`` quadrilateral unit square partitioned by `seam_split_partitioner`."""
    if comm.size > num_bands(n):
        pytest.skip(
            f"{comm.size} ranks but only {num_bands(n)} bands on a {n}x{n} grid"
        )
    cells, x = unit_square_block(comm, n)
    element = basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))
    partitioner = dolfinx.mesh.create_hybrid_cell_partitioner(
        _with_ghost_layer(seam_split_partitioner(n), n)
    )
    return dolfinx.mesh.create_mesh(
        comm,
        cells,
        element,
        x,
        partitioner=partitioner,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )


def doubly_periodic(x):
    """Left and bottom boundaries are merged into right and top: the square becomes a torus."""
    return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)


def doubly_periodic_map(x):
    v = x.copy()
    v[0] += np.isclose(x[0], 0.0) * 1.0
    v[1] += np.isclose(x[1], 0.0) * 1.0
    return v


# --------------------------------------------------------------------------- #
# The partition is what the docstring claims
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", GRID_SIZES)
def test_partition_splits_the_seam(n):
    """Both conditions the two bugs need are present in the input mesh.

    Without this, a change to `seam_split_partitioner` -- or a change in DOLFINx that moves
    entity ownership around -- could quietly turn the test below into one that exercises
    nothing, and it would still pass.
    """
    comm = MPI.COMM_WORLD
    mesh = split_seam_mesh(comm, n)
    tdim = mesh.topology.dim

    # The vertices the seam vertices get merged into, i.e. those on x = 1 or y = 1
    replacement = dolfinx.mesh.locate_entities(
        mesh, 0, lambda x: np.isclose(x[0], 1.0) | np.isclose(x[1], 1.0)
    )
    vertex_map = mesh.topology.index_map(0)
    replacement_global = set(
        vertex_map.local_to_global(np.asarray(replacement, dtype=np.int32)).tolist()
    )

    # Owners of the cells incident to each replacement vertex -- bug B needs at least two
    mesh.topology.create_connectivity(tdim, 0)
    c_to_v = mesh.topology.connectivity(tdim, 0)
    l2g = vertex_map.local_to_global(
        np.arange(vertex_map.size_local + vertex_map.num_ghosts, dtype=np.int32)
    )
    owners = {}
    for cell in range(mesh.topology.index_map(tdim).size_local):
        for v in l2g[c_to_v.links(cell)]:
            if int(v) in replacement_global:
                owners.setdefault(int(v), set()).add(comm.rank)
    merged = {}
    for chunk in comm.allgather(owners):
        for v, ranks in chunk.items():
            merged.setdefault(v, set()).update(ranks)
    if comm.size > 1:
        assert max(len(r) for r in merged.values()) > 1, (
            "no replacement vertex has its incident cells spread over several ranks:"
            " phase 3 cannot be exercised by this partition"
        )

    # Boundary facets on the seam that a rank holds but does not own -- bug A needs one.
    # `exterior_facet_indices` skips exactly these.
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    seam_facets = dolfinx.mesh.locate_entities(
        mesh, tdim - 1, lambda x: np.isclose(x[0], 1.0) | np.isclose(x[1], 1.0)
    )
    num_owned_facets = mesh.topology.index_map(tdim - 1).size_local
    ghosted = comm.allreduce(
        int(np.count_nonzero(seam_facets >= num_owned_facets)), MPI.SUM
    )
    if comm.size > 1:
        assert ghosted > 0, (
            "every seam boundary facet is owned by the rank that holds it:"
            " phase 1 cannot be exercised by this partition"
        )


# --------------------------------------------------------------------------- #
# The torus invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", GRID_SIZES)
def test_doubly_periodic_torus_has_no_boundary(n):
    """Every owned facet of the rebuilt mesh has two cells, and a periodic field is continuous.

    The facet count is the sharp assertion: a missing ghost cell leaves an owned facet with
    one incident cell whatever else looks right, and the vertex count, the volume and even
    the cell count are all still correct when it happens.

    The failures are rank-local -- one process cannot build the form while the others carry
    on -- so every count is reduced and every rank raises together. A rank that raised on its
    own would leave the others waiting in the next collective and the test would hang instead
    of failing.
    """
    comm = MPI.COMM_WORLD
    mesh = split_seam_mesh(comm, n)

    pm, _, _ = create_periodic_mesh(mesh, doubly_periodic, doubly_periodic_map)

    # Merging both seams removes one vertex per row and per column, leaving n * n
    assert pm.topology.index_map(0).size_global == n * n
    assert pm.topology.index_map(pm.topology.dim).size_global == n * n
    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=pm))), op=MPI.SUM
    )
    assert np.isclose(volume, 1.0), f"torus volume is {volume}"

    # A torus has no boundary, so no owned facet may have a single incident cell
    tdim = pm.topology.dim
    pm.topology.create_entities(tdim - 1)
    pm.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = pm.topology.connectivity(tdim - 1, tdim)
    num_owned_facets = pm.topology.index_map(tdim - 1).size_local
    cells_per_facet = np.diff(f_to_c.offsets)[:num_owned_facets]
    local_bad = int(np.count_nonzero(cells_per_facet != 2))
    num_bad = comm.allreduce(local_bad, op=MPI.SUM)
    assert num_bad == 0, (
        f"{num_bad} owned facets of the rebuilt periodic mesh do not have exactly two"
        " incident cells; the merged mesh is a torus and has no boundary, so a facet with"
        " one cell is a cell that was never shipped to this process"
    )

    V = dolfinx.fem.functionspace(pm, ("DG", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))

    # `dolfinx.fem.form` is where a missing ghost cell is reported, and only on the ranks
    # that are missing one. Reduce before raising: assembling below is collective.
    try:
        jump_form = dolfinx.fem.form(ufl.jump(u) ** 2 * ufl.dS)
        local_failure = 0
    except RuntimeError:
        jump_form = None
        local_failure = 1
    num_failed = comm.allreduce(local_failure, op=MPI.SUM)
    if num_failed:
        raise RuntimeError(
            f"{num_failed} of {comm.size} ranks cannot assemble an interior facet integral"
            " on the rebuilt periodic mesh: an interprocess facet is missing its ghost cell"
        )

    jump = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(jump_form), op=MPI.SUM))
    assert jump < 1e-10, f"{n}x{n} quad torus jumps by {jump:.3e}"
