# Tests for gmsh_periodic
# SPDX-License-Identifier: MIT

"""Serial tests for the ``$Periodic`` reader.

These build small gmsh models in process, so they need no mesh files and run in under a
second. Run them with::

    python3 -m pytest test_gmsh_periodic.py
"""

import inspect

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


# --------------------------------------------------------------------------- #
# the distributed path: gmsh pairs -> VertexCorrespondence -> periodic mesh
#
# Run under MPI to exercise it::
#
#     mpirun -n 3 python3 -m pytest test_gmsh_periodic.py
# --------------------------------------------------------------------------- #

from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import ufl  # noqa: E402

import script  # noqa: E402
from gmsh_periodic import (  # noqa: E402
    GmshPeriodicNodes,
    periodic_correspondence_from_nodes,
)


def _model_to_mesh(comm, rank, gdim):
    """``model_to_mesh`` with shared-facet ghosting, wherever the version wants it told.

    On 0.12 it is a ``model_to_mesh`` keyword; on 0.11 it goes through the partitioner.
    Getting this wrong does not fail here -- it fails much later, when an interior facet
    integral finds an interprocess facet with only one cell.
    """
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet
    if "ghost_mode" in inspect.signature(dolfinx.io.gmsh.model_to_mesh).parameters:
        kwargs = {"ghost_mode": ghost_mode}
    else:
        part_sig = inspect.signature(dolfinx.mesh.create_cell_partitioner)
        part_kwargs = (
            {"max_facet_to_cell_links": 2}
            if "max_facet_to_cell_links" in part_sig.parameters
            else {}
        )
        kwargs = {
            "partitioner": dolfinx.mesh.create_cell_partitioner(
                ghost_mode, **part_kwargs
            )
        }
    mesh_data = dolfinx.io.gmsh.model_to_mesh(
        gmsh.model, comm, rank, gdim=gdim, **kwargs
    )
    return getattr(mesh_data, "mesh", mesh_data)


def periodic_square(comm, res=1.0 / 5, directions=("x", "y"), low_is_slave=False):
    """A distributed unit square from gmsh, with the pairs read off its model.

    Args:
        comm: Communicator to distribute over.
        res: Target cell size.
        directions: Which directions to identify.
        low_is_slave: Replace the vertices at ``x=0``/``y=0`` rather than at 1. Matching
            the direction matters only when comparing against an `indicator`/`mapping`
            pair, which fixes which side is replaced.

    Returns:
        ``(mesh, pairs)``, with `pairs` meaningful on rank 0 only.
    """
    L = 1.0
    if comm.rank == 0:
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("periodic square")
        gmsh.model.occ.addRectangle(0, 0, 0, L, L)
        gmsh.model.occ.synchronize()
        # curve tags: 1 bottom, 2 right, 3 top, 4 left
        sign = -1.0 if low_is_slave else 1.0
        if "x" in directions:
            slave, master = ([4], [2]) if low_is_slave else ([2], [4])
            gmsh.model.mesh.setPeriodic(
                1,
                slave,
                master,
                [1, 0, 0, sign * L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            )
        if "y" in directions:
            slave, master = ([1], [3]) if low_is_slave else ([3], [1])
            gmsh.model.mesh.setPeriodic(
                1,
                slave,
                master,
                [1, 0, 0, 0, 0, 1, 0, sign * L, 0, 0, 1, 0, 0, 0, 0, 1],
            )
        # model_to_mesh refuses a model with no physical groups
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", res)
        gmsh.option.setNumber("Mesh.MeshSizeMax", res)
        gmsh.model.mesh.generate(2)
        pairs = extract_gmsh_periodic_nodes(gmsh.model)
    else:
        empty = np.zeros(0, dtype=np.int64)
        pairs = GmshPeriodicNodes(empty, empty, 0)

    mesh = _model_to_mesh(comm, 0, gdim=2)
    if comm.rank == 0:
        gmsh.finalize()
    return mesh, pairs


def torus_invariants(periodic_mesh):
    """``(num_vertices, volume, facets_without_two_cells, seam_jump)``, reduced."""
    comm = periodic_mesh.comm
    tdim = periodic_mesh.topology.dim
    num_vertices = periodic_mesh.topology.index_map(0).size_global
    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=periodic_mesh))),
        op=MPI.SUM,
    )
    periodic_mesh.topology.create_entities(tdim - 1)
    periodic_mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = periodic_mesh.topology.connectivity(tdim - 1, tdim)
    num_owned = periodic_mesh.topology.index_map(tdim - 1).size_local
    per_facet = (f_to_c.offsets[1:] - f_to_c.offsets[:-1])[:num_owned]
    bad = comm.allreduce(int(np.count_nonzero(per_facet != 2)), op=MPI.SUM)

    V = dolfinx.fem.functionspace(periodic_mesh, ("DG", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    jump = np.sqrt(
        comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.jump(u) ** 2 * ufl.dS)),
            op=MPI.SUM,
        )
    )
    return num_vertices, volume, bad, jump


def gathered_igi(mesh, vertices):
    """The input global indices of `vertices`, gathered and sorted, on every rank."""
    nodes = dolfinx.mesh.entities_to_geometry(mesh, 0, vertices).reshape(-1)
    igi = mesh.geometry.input_global_indices[nodes].astype(np.int64)
    everywhere = mesh.comm.allgather(igi)
    return np.unique(np.concatenate(everywhere)) if len(everywhere) else igi


def test_gmsh_path_builds_a_torus():
    """The whole point: `$Periodic` in, a mesh with no boundary out.

    A doubly periodic square is a torus, so every owned facet carries exactly two cells,
    and a periodic field does not jump across the merged seam. Counting vertices alone
    cannot tell a correct pairing from a wrong but plausible one; the jump can.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm)
    correspondence = periodic_correspondence_from_nodes(mesh, pairs)
    periodic_mesh, _, _ = script._build_periodic_mesh(mesh, correspondence)

    num_vertices, volume, bad, jump = torus_invariants(periodic_mesh)
    assert bad == 0, f"{bad} owned facet(s) do not carry two cells"
    assert np.isclose(volume, 1.0)
    assert jump < 1e-12, f"seam jumps by {jump:.3e}"
    # the mesh has 44 nodes and 11 of them are replaced, at this resolution
    assert num_vertices == 33


def test_gmsh_path_does_not_depend_on_the_partition():
    """The same answer at any rank count, which is what the numbers below pin.

    They are not free parameters: 44 gmsh nodes, 11 of them slaves. Run this file at 1, 2,
    3 and 4 ranks and the assertions are identical.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm)
    if comm.rank == 0:
        assert pairs.num_nodes_global == 44
        assert len(pairs.slave) == 11

    correspondence = periodic_correspondence_from_nodes(mesh, pairs)
    replaced = comm.allreduce(len(correspondence.indicator_vertices), op=MPI.SUM)
    assert replaced >= 11, (
        "a replaced vertex is missing from some process that holds it"
    )

    periodic_mesh, _, _ = script._build_periodic_mesh(mesh, correspondence)
    num_vertices, volume, bad, _ = torus_invariants(periodic_mesh)
    assert (num_vertices, bad) == (33, 0)
    assert np.isclose(volume, 1.0)


def test_gmsh_path_single_direction():
    """Periodic in x only: a cylinder, which still has the y=0 and y=1 boundaries."""
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm, directions=("x",))
    correspondence = periodic_correspondence_from_nodes(mesh, pairs)
    periodic_mesh, _, _ = script._build_periodic_mesh(mesh, correspondence)

    before = mesh.topology.index_map(0).size_global
    after = periodic_mesh.topology.index_map(0).size_global
    num_slaves = comm.bcast(len(pairs.slave) if comm.rank == 0 else None, root=0)
    assert after == before - num_slaves

    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=periodic_mesh))),
        op=MPI.SUM,
    )
    assert np.isclose(volume, 1.0)


def test_gmsh_path_replaces_the_same_vertices_as_the_geometric_path():
    """Both paths on the same mesh have to identify the same vertices.

    Compared in input global indices, not local ones, so the comparison says nothing about
    how the mesh happens to be partitioned. The gmsh model is built with the low side as
    the slave so that the two conventions agree on *which* side is replaced -- otherwise
    both are right and the sets are disjoint.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm, low_is_slave=True)

    def indicator(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)

    def mapping(x):
        v = x.copy()
        v[0] += np.isclose(x[0], 0.0) * 1.0
        v[1] += np.isclose(x[1], 0.0) * 1.0
        return v

    geometric = script._match_vertices_geometric(mesh, indicator, mapping)
    from_gmsh = periodic_correspondence_from_nodes(mesh, pairs)

    assert np.array_equal(
        gathered_igi(mesh, geometric.indicator_vertices),
        gathered_igi(mesh, from_gmsh.indicator_vertices),
    ), "the two paths replace different vertices"
    assert np.array_equal(
        np.sort(geometric.indicator_facets), np.sort(from_gmsh.indicator_facets)
    ), "the two paths disagree on the seam facets"

    # and the meshes they build agree on the invariants
    from_geometric, _, _ = script._build_periodic_mesh(mesh, geometric)
    from_pairs, _, _ = script._build_periodic_mesh(mesh, from_gmsh)
    assert torus_invariants(from_geometric)[:3] == torus_invariants(from_pairs)[:3]
