# Tests for transfer_meshtags_to_periodic_mesh
# SPDX-License-Identifier: MIT

"""Tags are checked by where they land, not by how many survive.

A transfer that loses the entity/value association keeps every count intact, so each test
here re-derives the value from the tagged entity's own midpoint on the periodic mesh and
asserts it is the value that entity should carry. Counting is the weaker half of each
test, kept because it is what catches an entity being dropped.

The one rule with real content is which entities are dropped: an entity survives iff at
least one of its vertices survives, so a facet lying *on* the seam disappears -- it has
been merged into its partner -- while a facet merely touching the seam at one vertex does
not.

Run serially, or under MPI::

    python3 -m pytest test_meshtags.py
    mpirun -n 3 python3 -m pytest test_meshtags.py
"""

import numpy as np
import pytest
from mpi4py import MPI

import dolfinx

from scifem.periodic.mesh import create_periodic_mesh
from scifem.periodic.transfer import transfer_meshtags_to_periodic_mesh


def unit_square(n=8):
    return dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )


def x_periodic():
    """Replace the vertices at x=0 with those at x=1."""

    def indicator(x):
        return np.isclose(x[0], 0.0)

    def mapping(x):
        values = x.copy()
        values[0] += 1.0
        return values

    return indicator, mapping


def owned(mesh, dim, entities):
    """The entities of `entities` this process owns, so a global count is not double-counted."""
    entities = np.asarray(entities, dtype=np.int32)
    return entities[entities < mesh.topology.index_map(dim).size_local]


def count(mesh, tags, value):
    """How many entities carry `value`, over the whole communicator."""
    marked = tags.indices[tags.values == value]
    return mesh.comm.allreduce(len(owned(mesh, tags.dim, marked)), op=MPI.SUM)


def midpoints(mesh, dim, entities):
    # `compute_midpoints` reaches for the entity-to-cell connectivity, which is not there
    # by default for anything below the cell dimension.
    mesh.topology.create_entities(dim)
    mesh.topology.create_connectivity(dim, mesh.topology.dim)
    entities = np.asarray(entities, dtype=np.int32)
    return dolfinx.mesh.compute_midpoints(mesh, dim, entities)


def tag_by(mesh, dim, value_of, where=None):
    """Tag the entities `where` selects with ``value_of(midpoint)``, one value per entity.

    Args:
        mesh: The mesh to tag.
        dim: Dimension of the entities to tag.
        value_of: Midpoints as ``(n, 3)`` to an integer value per entity.
        where: Locator for the entities to tag; all of them if omitted.
    """
    mesh.topology.create_entities(dim)
    if where is None:
        imap = mesh.topology.index_map(dim)
        entities = np.arange(imap.size_local, dtype=np.int32)
    else:
        entities = owned(mesh, dim, dolfinx.mesh.locate_entities(mesh, dim, where))
    values = np.asarray(value_of(midpoints(mesh, dim, entities)), dtype=np.int32)
    order = np.argsort(entities)
    return dolfinx.mesh.meshtags(mesh, dim, entities[order], values[order])


def band(x):
    """1 below y=1/2, 2 above -- a value that can be re-derived from a midpoint."""
    return np.where(x[:, 1] < 0.5, 1, 2)


# --------------------------------------------------------------------------- #
# cells: nothing is ever dropped
# --------------------------------------------------------------------------- #


def test_cell_tags_are_preserved():
    """The ``dim == tdim`` branch skips the filter entirely, so every cell has to survive.

    Cell ownership does not change in the rebuild, so this is the transfer at its
    simplest: the same cells, the same values, only renumbered.
    """
    n = 8
    mesh = unit_square(n)
    tags = tag_by(mesh, mesh.topology.dim, band)
    pm, replaced, _ = create_periodic_mesh(mesh, *x_periodic())
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)

    assert moved.dim == mesh.topology.dim
    total = mesh.comm.allreduce(len(owned(pm, moved.dim, moved.indices)), op=MPI.SUM)
    assert total == pm.topology.index_map(pm.topology.dim).size_global
    assert count(pm, moved, 1) == n * n
    assert count(pm, moved, 2) == n * n


def test_cell_tag_values_follow_their_cells():
    """The counts above are preserved by any permutation; this is not."""
    mesh = unit_square(8)
    tags = tag_by(mesh, mesh.topology.dim, band)
    pm, replaced, _ = create_periodic_mesh(mesh, *x_periodic())
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)

    here = owned(pm, moved.dim, moved.indices)
    values = moved.values[np.isin(moved.indices, here)]
    assert np.array_equal(values, band(midpoints(pm, moved.dim, here)))


# --------------------------------------------------------------------------- #
# facets: dropped iff every vertex is replaced
# --------------------------------------------------------------------------- #


def test_facet_tags_off_the_seam_are_preserved():
    """y=0 and y=1 are untouched by an x-periodic identification.

    The two facets at the corners have one replaced vertex each and are kept on the rule
    above, so the count is exact rather than approximate.
    """
    n = 8
    mesh = unit_square(n)
    tdim = mesh.topology.dim
    tags = tag_by(
        mesh,
        tdim - 1,
        lambda m: np.where(m[:, 1] < 0.5, 1, 2),
        where=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
    )
    assert count(mesh, tags, 1) == n and count(mesh, tags, 2) == n

    pm, replaced, _ = create_periodic_mesh(mesh, *x_periodic())
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)
    assert count(pm, moved, 1) == n
    assert count(pm, moved, 2) == n


def test_facet_tag_values_follow_their_facets():
    """Where each tagged facet ended up, not just how many did."""
    mesh = unit_square(8)
    tdim = mesh.topology.dim
    tags = tag_by(
        mesh,
        tdim - 1,
        lambda m: np.where(m[:, 1] < 0.5, 1, 2),
        where=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
    )
    pm, replaced, _ = create_periodic_mesh(mesh, *x_periodic())
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)

    here = owned(pm, moved.dim, moved.indices)
    values = moved.values[np.isin(moved.indices, here)]
    mid = midpoints(pm, moved.dim, here)
    assert np.all(np.isclose(mid[:, 1], 0.0) | np.isclose(mid[:, 1], 1.0))
    assert np.array_equal(values, np.where(mid[:, 1] < 0.5, 1, 2))


def test_facets_on_the_seam_are_dropped():
    """The replaced side of the seam has no image: it *is* the other side now.

    Every vertex of an x=0 facet is replaced, so the rule drops it. Keeping it would
    address it by input global indices that the periodic mesh no longer has.
    """
    n = 8
    mesh = unit_square(n)
    tdim = mesh.topology.dim
    tags = tag_by(
        mesh,
        tdim - 1,
        lambda m: np.full(len(m), 7),
        where=lambda x: np.isclose(x[0], 0.0),
    )
    assert count(mesh, tags, 7) == n

    pm, replaced, _ = create_periodic_mesh(mesh, *x_periodic())
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)
    assert count(pm, moved, 7) == 0


def test_the_surviving_side_of_the_seam_is_kept():
    """The mirror of the test above, and the one that would fail if the rule inverted.

    Its midpoint is at x=0 *or* x=1, and which one is not a property of the tag. The two
    facets are one facet now, and it has two geometric representatives -- the geometry is
    deliberately torn across the seam, since the cells still sit where they sit. What
    `compute_midpoints` reports is whichever incident cell the connectivity lists first.
    """
    n = 8
    mesh = unit_square(n)
    tdim = mesh.topology.dim
    tags = tag_by(
        mesh,
        tdim - 1,
        lambda m: np.full(len(m), 9),
        where=lambda x: np.isclose(x[0], 1.0),
    )
    pm, replaced, _ = create_periodic_mesh(mesh, *x_periodic())
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)

    assert count(pm, moved, 9) == n
    here = owned(pm, moved.dim, moved.indices)
    x = midpoints(pm, moved.dim, here)[:, 0]
    assert np.all(np.isclose(x, 0.0) | np.isclose(x, 1.0))


def test_a_facet_touching_the_seam_at_one_vertex_is_kept():
    """The boundary case the rule is written for, pinned on its own.

    The bottom-left facet of an x-periodic square has its (0, 0) vertex replaced and its
    (1/n, 0) vertex retained. It survives, and it survives as a facet of the cell it
    belonged to.
    """
    n = 8
    mesh = unit_square(n)
    tdim = mesh.topology.dim
    tags = tag_by(
        mesh,
        tdim - 1,
        lambda m: np.full(len(m), 3),
        where=lambda x: np.isclose(x[1], 0.0) & (x[0] < 1.0 / n + 1e-12),
    )
    assert count(mesh, tags, 3) == 1

    pm, replaced, _ = create_periodic_mesh(mesh, *x_periodic())
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)
    assert count(pm, moved, 3) == 1
    here = owned(pm, moved.dim, moved.indices)
    if len(here):
        mid = midpoints(pm, moved.dim, here)
        assert np.allclose(mid[0], [0.5 / n, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# doubly periodic, where a facet can be dropped from either direction
# --------------------------------------------------------------------------- #


def test_doubly_periodic_drops_both_seams_and_keeps_the_interior():
    """With no boundary left, only tags on interior facets can survive."""
    n = 6
    mesh = unit_square(n)
    tdim = mesh.topology.dim

    def indicator(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)

    def mapping(x):
        values = x.copy()
        values[0] += np.isclose(x[0], 0.0) * 1.0
        values[1] += np.isclose(x[1], 0.0) * 1.0
        return values

    tags = tag_by(
        mesh,
        tdim - 1,
        lambda m: np.where(np.isclose(m[:, 0], 0.0), 1, 2),
        where=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0),
    )
    assert count(mesh, tags, 1) == n and count(mesh, tags, 2) == n

    pm, replaced, _ = create_periodic_mesh(mesh, indicator, mapping)
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)
    assert count(pm, moved, 1) == 0
    assert count(pm, moved, 2) == 0


@pytest.mark.parametrize("n", [4, 8])
def test_interior_facet_tags_survive_a_double_identification(n):
    """A band of interior facets is untouched by either seam, at either mesh size."""
    mesh = unit_square(n)
    tdim = mesh.topology.dim

    def indicator(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)

    def mapping(x):
        values = x.copy()
        values[0] += np.isclose(x[0], 0.0) * 1.0
        values[1] += np.isclose(x[1], 0.0) * 1.0
        return values

    tags = tag_by(
        mesh,
        tdim - 1,
        lambda m: np.full(len(m), 5),
        where=lambda x: np.isclose(x[1], 0.5),
    )
    before = count(mesh, tags, 5)
    assert before > 0

    pm, replaced, _ = create_periodic_mesh(mesh, indicator, mapping)
    moved = transfer_meshtags_to_periodic_mesh(mesh, pm, replaced, tags)
    assert count(pm, moved, 5) == before
    here = owned(pm, moved.dim, moved.indices)
    assert np.allclose(midpoints(pm, moved.dim, here)[:, 1], 0.5)
