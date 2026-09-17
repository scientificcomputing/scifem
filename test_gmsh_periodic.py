# Tests for gmsh_periodic
# SPDX-License-Identifier: MIT

"""Serial tests for the ``$Periodic`` reader.

These build small gmsh models in process, so they need no mesh files and run in under a
second. Run them with::

    python3 -m pytest test_gmsh_periodic.py
"""

import gmsh
import numpy as np
import pytest

from gmsh_periodic import extract_gmsh_periodic_nodes


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield gmsh.model
    finally:
        gmsh.finalize()


def _rectangle(model, L=1.0, res=1.0 / 3.0, directions=("x", "y"), order=1):
    """A meshed unit rectangle, periodic in the requested directions.

    The right curve is the slave of the left one, the top of the bottom, matching how a
    caller would usually write it. Curve tags of ``addRectangle`` are 1 bottom, 2 right,
    3 top, 4 left.
    """
    model.add("rect")
    model.occ.addRectangle(0, 0, 0, L, L)
    model.occ.synchronize()
    if "x" in directions:
        model.mesh.setPeriodic(
            1, [2], [4], [1, 0, 0, L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        )
    if "y" in directions:
        model.mesh.setPeriodic(
            1, [3], [1], [1, 0, 0, 0, 0, 1, 0, L, 0, 0, 1, 0, 0, 0, 0, 1]
        )
    gmsh.option.setNumber("Mesh.MeshSizeMin", res)
    gmsh.option.setNumber("Mesh.MeshSizeMax", res)
    model.mesh.generate(2)
    if order > 1:
        model.mesh.setOrder(order)
    return model


def _coords(model, tags_zero_based):
    """Coordinates of 0-based node tags."""
    return np.array([model.mesh.getNode(int(t) + 1)[0] for t in tags_zero_based])


def test_no_periodicity_gives_no_pairs(gmsh_session):
    model = _rectangle(gmsh_session, directions=())
    pairs = extract_gmsh_periodic_nodes(model)
    assert len(pairs.slave) == 0
    assert len(pairs.master) == 0
    assert pairs.num_nodes_global > 0


def test_single_direction_pairs_opposite_sides(gmsh_session):
    model = _rectangle(gmsh_session, directions=("x",))
    pairs = extract_gmsh_periodic_nodes(model)

    assert len(pairs.slave) > 0
    xs = _coords(model, pairs.slave)
    xm = _coords(model, pairs.master)
    assert np.allclose(xs[:, 0], 1.0), "slaves are not on x=1"
    assert np.allclose(xm[:, 0], 0.0), "masters are not on x=0"
    assert np.allclose(xs[:, 1], xm[:, 1]), "pairing does not preserve y"


def test_every_master_is_a_root(gmsh_session):
    """The contract the rebuild depends on: a master is never itself replaced."""
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)
    assert not np.isin(pairs.master, pairs.slave).any()


def test_slaves_are_unique_and_sorted(gmsh_session):
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)
    assert len(np.unique(pairs.slave)) == len(pairs.slave)
    assert (np.diff(pairs.slave) > 0).all()
    assert len(pairs.slave) == len(pairs.master)


def test_corner_resolves_through_the_chain(gmsh_session):
    """(1,1) is paired with (0,1) and with (1,0), and must come out as (0,0).

    This is the case that makes the pairs a relation rather than a function, and the one
    that fails if the chain is followed only one hop.
    """
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)

    xs = _coords(model, pairs.slave)
    xm = _coords(model, pairs.master)
    corner = np.flatnonzero(np.isclose(xs[:, 0], 1.0) & np.isclose(xs[:, 1], 1.0))
    assert len(corner) == 1, "the (1,1) corner is not a slave exactly once"
    assert np.allclose(xm[corner[0]][:2], [0.0, 0.0]), (
        f"corner resolved to {xm[corner[0]][:2]} instead of the opposite corner"
    )


def test_doubly_periodic_pairing_is_geometrically_consistent(gmsh_session):
    """Every slave differs from its master by whole periods in x and y."""
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)

    offset = _coords(model, pairs.slave) - _coords(model, pairs.master)
    assert np.allclose(offset, np.round(offset)), (
        "a pair is not separated by a whole number of periods"
    )
    assert np.allclose(offset[:, 2], 0.0)


def test_node_count_is_the_gmsh_count(gmsh_session):
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)
    tags, _, _ = model.mesh.getNodes()
    assert pairs.num_nodes_global == int(np.asarray(tags).max())
    assert pairs.slave.max() < pairs.num_nodes_global
    assert pairs.master.max() < pairs.num_nodes_global


def test_high_order_flag_adds_only_non_vertex_nodes(gmsh_session):
    """The flag adds pairs, and the *vertex* pairs are the same either way.

    That second half is what licenses the default being ``False``: the correspondence is
    between vertices, and those come out identical.
    """
    model = _rectangle(gmsh_session, directions=("x", "y"), order=2)
    without = extract_gmsh_periodic_nodes(model, include_high_order=False)
    with_ = extract_gmsh_periodic_nodes(model, include_high_order=True)

    assert len(with_.slave) > len(without.slave), "P2 added no extra nodes"
    keep = np.isin(with_.slave, without.slave)
    assert np.array_equal(with_.slave[keep], without.slave)
    assert np.array_equal(with_.master[keep], without.master)


def test_affine_violation_is_reported(monkeypatch, gmsh_session):
    """A transform that does not match the node positions has to raise, not resolve.

    The transform is corrupted on the way out of gmsh rather than by calling
    ``setPeriodic`` again: gmsh keeps the transform recorded when the entity was meshed
    and ignores a later one, so the model cannot be made to lie about it directly.
    """
    model = _rectangle(gmsh_session, directions=("x",))
    extract_gmsh_periodic_nodes(model)  # the honest model passes

    real = model.mesh.getPeriodicNodes

    def wrong_affine(dim, tag, include_high_order=False):
        master_tag, nodes, masters, affine = real(dim, tag, include_high_order)
        if master_tag != tag and len(affine) == 16:
            affine = list(affine)
            affine[3] = 2.0  # claim a translation of 2 where the geometry has 1
        return master_tag, nodes, masters, affine

    monkeypatch.setattr(model.mesh, "getPeriodicNodes", wrong_affine)
    with pytest.raises(RuntimeError, match="affine transform"):
        extract_gmsh_periodic_nodes(model)


def test_missing_affine_is_tolerated(monkeypatch, gmsh_session):
    """gmsh stores no transform for some entities; that is not an error."""
    model = _rectangle(gmsh_session, directions=("x", "y"))
    expected = extract_gmsh_periodic_nodes(model)

    real = model.mesh.getPeriodicNodes

    def no_affine(dim, tag, include_high_order=False):
        master_tag, nodes, masters, _ = real(dim, tag, include_high_order)
        return master_tag, nodes, masters, []

    monkeypatch.setattr(model.mesh, "getPeriodicNodes", no_affine)
    pairs = extract_gmsh_periodic_nodes(model)
    assert np.array_equal(pairs.slave, expected.slave)
    assert np.array_equal(pairs.master, expected.master)


def test_cycle_is_rejected(monkeypatch, gmsh_session):
    """Pairs that never reach a root must stop, not spin."""
    model = _rectangle(gmsh_session, directions=("x",))
    real = model.mesh.getPeriodicNodes

    def cyclic(dim, tag, include_high_order=False):
        master_tag, nodes, masters, affine = real(dim, tag, include_high_order)
        if master_tag != tag and len(nodes):
            # send the pairing back on itself, so nothing is ever a root
            return (
                master_tag,
                list(nodes) + list(masters),
                list(masters) + list(nodes),
                [],
            )
        return master_tag, nodes, masters, affine

    monkeypatch.setattr(model.mesh, "getPeriodicNodes", cyclic)
    with pytest.raises(RuntimeError, match="cycle|do not end|different roots"):
        extract_gmsh_periodic_nodes(model)


def test_inconsistent_pairs_are_rejected(monkeypatch, gmsh_session):
    """Two routes out of one node that disagree on the root must raise."""
    model = _rectangle(gmsh_session, directions=("x",))
    real = model.mesh.getPeriodicNodes

    def contradictory(dim, tag, include_high_order=False):
        master_tag, nodes, masters, affine = real(dim, tag, include_high_order)
        if master_tag != tag and len(nodes) >= 2:
            # give the first slave a second, unrelated master that is not a slave
            extra_master = max(masters) + 1 if max(masters) + 1 not in nodes else None
            if extra_master is not None:
                return (
                    master_tag,
                    list(nodes) + [nodes[0]],
                    list(masters) + [extra_master],
                    [],
                )
        return master_tag, nodes, masters, affine

    monkeypatch.setattr(model.mesh, "getPeriodicNodes", contradictory)
    with pytest.raises(RuntimeError, match="different roots"):
        extract_gmsh_periodic_nodes(model)
