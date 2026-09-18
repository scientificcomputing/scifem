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

from scifem.periodic.gmsh import extract_gmsh_periodic_nodes
import scifem.periodic.mesh


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

    The right curve is replaced by the left one, the top by the bottom, matching how a
    caller would usually write it. Curve tags of ``addRectangle`` are 1 bottom, 2 right,
    3 top, 4 left.
    """
    model.add("rect")
    model.occ.addRectangle(0, 0, 0, L, L)
    model.occ.synchronize()
    if "x" in directions:
        model.mesh.setPeriodic(1, [2], [4], [1, 0, 0, L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    if "y" in directions:
        model.mesh.setPeriodic(1, [3], [1], [1, 0, 0, 0, 0, 1, 0, L, 0, 0, 1, 0, 0, 0, 0, 1])
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
    assert len(pairs.replaced) == 0
    assert len(pairs.partner) == 0
    assert pairs.num_nodes_global > 0


def test_single_direction_pairs_opposite_sides(gmsh_session):
    model = _rectangle(gmsh_session, directions=("x",))
    pairs = extract_gmsh_periodic_nodes(model)

    assert len(pairs.replaced) > 0
    xs = _coords(model, pairs.replaced)
    xm = _coords(model, pairs.partner)
    assert np.allclose(xs[:, 0], 1.0), "the replaced nodes are not on x=1"
    assert np.allclose(xm[:, 0], 0.0), "the partner nodes are not on x=0"
    assert np.allclose(xs[:, 1], xm[:, 1]), "pairing does not preserve y"


def test_every_partner_is_a_root(gmsh_session):
    """The contract the rebuild depends on: a partner is never itself replaced."""
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)
    assert not np.isin(pairs.partner, pairs.replaced).any()


def test_replaced_nodes_are_unique_and_sorted(gmsh_session):
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)
    assert len(np.unique(pairs.replaced)) == len(pairs.replaced)
    assert (np.diff(pairs.replaced) > 0).all()
    assert len(pairs.replaced) == len(pairs.partner)


def test_corner_resolves_through_the_chain(gmsh_session):
    """(1,1) is paired with (0,1) and with (1,0), and must come out as (0,0).

    This is the case that makes the pairs a relation rather than a function, and the one
    that fails if the chain is followed only one hop.
    """
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)

    xs = _coords(model, pairs.replaced)
    xm = _coords(model, pairs.partner)
    corner = np.flatnonzero(np.isclose(xs[:, 0], 1.0) & np.isclose(xs[:, 1], 1.0))
    assert len(corner) == 1, "the (1,1) corner is not replaced exactly once"
    assert np.allclose(xm[corner[0]][:2], [0.0, 0.0]), (
        f"corner resolved to {xm[corner[0]][:2]} instead of the opposite corner"
    )


def test_doubly_periodic_pairing_is_geometrically_consistent(gmsh_session):
    """Every replaced differs from its partner by whole periods in x and y."""
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)

    offset = _coords(model, pairs.replaced) - _coords(model, pairs.partner)
    assert np.allclose(offset, np.round(offset)), (
        "a pair is not separated by a whole number of periods"
    )
    assert np.allclose(offset[:, 2], 0.0)


def test_node_count_is_the_gmsh_count(gmsh_session):
    model = _rectangle(gmsh_session, directions=("x", "y"))
    pairs = extract_gmsh_periodic_nodes(model)
    tags, _, _ = model.mesh.getNodes()
    assert pairs.num_nodes_global == int(np.asarray(tags).max())
    assert pairs.replaced.max() < pairs.num_nodes_global
    assert pairs.partner.max() < pairs.num_nodes_global


def test_high_order_flag_adds_only_non_vertex_nodes(gmsh_session):
    """The flag adds pairs, and the *vertex* pairs are the same either way.

    That second half is what licenses the default being ``False``: the correspondence is
    between vertices, and those come out identical.
    """
    model = _rectangle(gmsh_session, directions=("x", "y"), order=2)
    without = extract_gmsh_periodic_nodes(model, include_high_order=False)
    with_ = extract_gmsh_periodic_nodes(model, include_high_order=True)

    assert len(with_.replaced) > len(without.replaced), "P2 added no extra nodes"
    keep = np.isin(with_.replaced, without.replaced)
    assert np.array_equal(with_.replaced[keep], without.replaced)
    assert np.array_equal(with_.partner[keep], without.partner)


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
        partner_tag, nodes, partner_nodes, affine = real(dim, tag, include_high_order)
        if partner_tag != tag and len(affine) == 16:
            affine = list(affine)
            affine[3] = 2.0  # claim a translation of 2 where the geometry has 1
        return partner_tag, nodes, partner_nodes, affine

    monkeypatch.setattr(model.mesh, "getPeriodicNodes", wrong_affine)
    with pytest.raises(RuntimeError, match="affine transform"):
        extract_gmsh_periodic_nodes(model)

    # `tol=None` turns the check off, and the pairs themselves are unaffected by it: the
    # transform is only ever read to be verified, never to derive anything.
    monkeypatch.setattr(model.mesh, "getPeriodicNodes", real)
    honest = extract_gmsh_periodic_nodes(model)
    monkeypatch.setattr(model.mesh, "getPeriodicNodes", wrong_affine)
    unchecked = extract_gmsh_periodic_nodes(model, tol=None)
    assert np.array_equal(unchecked.replaced, honest.replaced)
    assert np.array_equal(unchecked.partner, honest.partner)


def test_missing_affine_is_tolerated(monkeypatch, gmsh_session):
    """gmsh stores no transform for some entities; that is not an error."""
    model = _rectangle(gmsh_session, directions=("x", "y"))
    expected = extract_gmsh_periodic_nodes(model)

    real = model.mesh.getPeriodicNodes

    def no_affine(dim, tag, include_high_order=False):
        partner_tag, nodes, partner_nodes, _ = real(dim, tag, include_high_order)
        return partner_tag, nodes, partner_nodes, []

    monkeypatch.setattr(model.mesh, "getPeriodicNodes", no_affine)
    pairs = extract_gmsh_periodic_nodes(model)
    assert np.array_equal(pairs.replaced, expected.replaced)
    assert np.array_equal(pairs.partner, expected.partner)


def test_cycle_is_rejected(monkeypatch, gmsh_session):
    """Pairs that never reach a root must stop, not spin."""
    model = _rectangle(gmsh_session, directions=("x",))
    real = model.mesh.getPeriodicNodes

    def cyclic(dim, tag, include_high_order=False):
        partner_tag, nodes, partner_nodes, affine = real(dim, tag, include_high_order)
        if partner_tag != tag and len(nodes):
            # send the pairing back on itself, so nothing is ever a root
            return (
                partner_tag,
                list(nodes) + list(partner_nodes),
                list(partner_nodes) + list(nodes),
                [],
            )
        return partner_tag, nodes, partner_nodes, affine

    monkeypatch.setattr(model.mesh, "getPeriodicNodes", cyclic)
    with pytest.raises(RuntimeError, match="cycle|do not end|different roots"):
        extract_gmsh_periodic_nodes(model)


def test_inconsistent_pairs_are_rejected(monkeypatch, gmsh_session):
    """Two routes out of one node that disagree on the root must raise."""
    model = _rectangle(gmsh_session, directions=("x",))
    real = model.mesh.getPeriodicNodes

    def contradictory(dim, tag, include_high_order=False):
        partner_tag, nodes, partner_nodes, affine = real(dim, tag, include_high_order)
        if partner_tag != tag and len(nodes) >= 2:
            # give the first replaced node a second, unrelated partner of its own
            extra_partner = max(partner_nodes) + 1 if max(partner_nodes) + 1 not in nodes else None
            if extra_partner is not None:
                return (
                    partner_tag,
                    list(nodes) + [nodes[0]],
                    list(partner_nodes) + [extra_partner],
                    [],
                )
        return partner_tag, nodes, partner_nodes, affine

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

from scifem.periodic.topological_search import periodic_correspondence_from_nodes

from scifem.periodic.gmsh import (  # noqa: E402
    PeriodicNodes,
    read_periodic_mesh_from_msh,
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
        kwargs = {"partitioner": dolfinx.mesh.create_cell_partitioner(ghost_mode, **part_kwargs)}
    mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, comm, rank, gdim=gdim, **kwargs)
    return getattr(mesh_data, "mesh", mesh_data)


def periodic_square(comm, res=1.0 / 5, directions=("x", "y"), low_is_replaced=False, order=1):
    """A distributed unit square from gmsh, with the pairs read off its model.

    Args:
        comm: Communicator to distribute over.
        res: Target cell size.
        directions: Which directions to identify.
        low_is_replaced: Replace the vertices at ``x=0``/``y=0`` rather than at 1. Matching
            the direction matters only when comparing against an `indicator`/`mapping`
            pair, which fixes which side is replaced.
        order: Geometry degree. Above 1 the mesh gains nodes that are not vertices, which
            the reader has to leave out of the correspondence.

    Returns:
        ``(mesh, pairs)``, with `pairs` meaningful on rank 0 only.
    """
    L = 1.0
    started_here = False
    if comm.rank == 0:
        if not gmsh.isInitialized():
            gmsh.initialize()
            started_here = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("periodic square")
        gmsh.model.occ.addRectangle(0, 0, 0, L, L)
        gmsh.model.occ.synchronize()
        # curve tags: 1 bottom, 2 right, 3 top, 4 left
        sign = -1.0 if low_is_replaced else 1.0
        if "x" in directions:
            replaced, partner = ([4], [2]) if low_is_replaced else ([2], [4])
            gmsh.model.mesh.setPeriodic(
                1,
                replaced,
                partner,
                [1, 0, 0, sign * L, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            )
        if "y" in directions:
            replaced, partner = ([1], [3]) if low_is_replaced else ([3], [1])
            gmsh.model.mesh.setPeriodic(
                1,
                replaced,
                partner,
                [1, 0, 0, 0, 0, 1, 0, sign * L, 0, 0, 1, 0, 0, 0, 0, 1],
            )
        # model_to_mesh refuses a model with no physical groups
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", res)
        gmsh.option.setNumber("Mesh.MeshSizeMax", res)
        gmsh.model.mesh.generate(2)
        if order > 1:
            gmsh.model.mesh.setOrder(order)
        pairs = extract_gmsh_periodic_nodes(gmsh.model)
    else:
        pairs = PeriodicNodes()

    mesh = _model_to_mesh(comm, 0, gdim=2)
    if started_here:
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
    # Periodic in every direction means no boundary: the mesh is a closed manifold, so
    # every facet has exactly two cells. The difference of consecutive offsets is the
    # number of cells at each facet, straight off the adjacency list. Owned facets only --
    # a shared facet lives on several ranks, so counting all of them would count it once
    # per holder, and a ghost facet need not carry both of its cells in the first place.
    #
    # Since `check_seam_is_manifold` now raises inside the rebuild, more than two cannot
    # reach here: what this counts in practice is facets left with *one*. That is either a
    # seam the pairing never glued, or one glued to a cell that was never shipped -- the
    # failure the session spent its time on. It is the half the seam jump below cannot
    # see, because a jump can only be measured across facets that exist.
    periodic_mesh.topology.create_entities(tdim - 1)
    periodic_mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = periodic_mesh.topology.connectivity(tdim - 1, tdim)
    num_owned = periodic_mesh.topology.index_map(tdim - 1).size_local
    per_facet = np.diff(f_to_c.offsets)[:num_owned]
    bad = comm.allreduce(int(np.count_nonzero(per_facet != 2)), op=MPI.SUM)

    V = dolfinx.fem.functionspace(periodic_mesh, ("DG", 1))
    u = dolfinx.fem.Function(V)

    def periodic_field(x):
        """Periodic under unit translation in every direction the mesh has."""
        value = np.ones(x.shape[1])
        for d in range(periodic_mesh.geometry.dim):
            value = value * np.cos(2 * np.pi * x[d])
        return value

    u.interpolate(periodic_field)

    # Building the dS form is where a missing ghost cell surfaces, and it fails on the
    # affected rank only. Reduce before acting on it: a bare raise here would leave the
    # other ranks blocked in the assembly below, so the run hangs instead of reporting.
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
            " on the rebuilt mesh: an interprocess facet is missing its ghost cell"
        )

    jump = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(jump_form), op=MPI.SUM))
    return num_vertices, volume, bad, jump


def gathered_igi(mesh, vertices):
    """The input global indices of `vertices`, gathered and sorted, on every rank."""
    nodes = dolfinx.mesh.entities_to_geometry(mesh, 0, vertices).reshape(-1)
    igi = mesh.geometry.input_global_indices[nodes].astype(np.int64)
    everywhere = mesh.comm.allgather(igi)
    return np.unique(np.concatenate(everywhere)) if len(everywhere) else igi


def assert_everywhere(comm, holds, message):
    """Assert across ranks, so a disagreement on one of them reports instead of hanging.

    A bare assert in an MPI test is one-sided: the rank that fails leaves the collectives,
    and the others block in the next one, so the run hangs with no usable output.
    """
    failed = comm.allreduce(0 if holds else 1, op=MPI.SUM)
    assert failed == 0, f"{message} (on {failed} of {comm.size} ranks)"


def test_gmsh_path_builds_a_torus():
    """The whole point: `$Periodic` in, a mesh with no boundary out.

    A doubly periodic square is a torus, so every owned facet carries exactly two cells,
    and a periodic field does not jump across the merged seam. Counting vertices alone
    cannot tell a correct pairing from a wrong but plausible one; the jump can.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm)
    correspondence = periodic_correspondence_from_nodes(mesh, pairs)
    periodic_mesh, _, _ = scifem.periodic.mesh._build_periodic_mesh(mesh, correspondence)

    num_vertices, volume, bad, jump = torus_invariants(periodic_mesh)
    assert bad == 0, f"{bad} owned facet(s) do not carry two cells"
    assert np.isclose(volume, 1.0)
    assert jump < 1e-12, f"seam jumps by {jump:.3e}"
    # the mesh has 44 nodes and 11 of them are replaced, at this resolution
    assert num_vertices == 33


def test_gmsh_path_does_not_depend_on_the_partition():
    """The same answer at any rank count, which is what the numbers below pin.

    They are not free parameters: 44 gmsh nodes, 11 of them replaced_nodes. Run this file at 1, 2,
    3 and 4 ranks and the assertions are identical.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm)
    if comm.rank == 0:
        assert pairs.num_nodes_global == 44
        assert len(pairs.replaced) == 11

    correspondence = periodic_correspondence_from_nodes(mesh, pairs)
    replaced = comm.allreduce(len(correspondence.indicator_vertices), op=MPI.SUM)
    assert replaced >= 11, "a replaced vertex is missing from some process that holds it"

    periodic_mesh, _, _ = scifem.periodic.mesh._build_periodic_mesh(mesh, correspondence)
    num_vertices, volume, bad, _ = torus_invariants(periodic_mesh)
    assert (num_vertices, bad) == (33, 0)
    assert np.isclose(volume, 1.0)


def test_gmsh_path_single_direction():
    """Periodic in x only: a cylinder, which still has the y=0 and y=1 boundaries."""
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm, directions=("x",))
    correspondence = periodic_correspondence_from_nodes(mesh, pairs)
    periodic_mesh, _, _ = scifem.periodic.mesh._build_periodic_mesh(mesh, correspondence)

    before = mesh.topology.index_map(0).size_global
    after = periodic_mesh.topology.index_map(0).size_global
    num_replaced = comm.bcast(len(pairs.replaced) if comm.rank == 0 else None, root=0)
    assert after == before - num_replaced

    volume = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=periodic_mesh))),
        op=MPI.SUM,
    )
    assert np.isclose(volume, 1.0)


def test_gmsh_path_replaces_the_same_vertices_as_the_geometric_path():
    """Both paths on the same mesh have to identify the same vertices.

    Compared in input global indices, not local ones, so the comparison says nothing about
    how the mesh happens to be partitioned. The gmsh model is built with the low side as
    the replaced so that the two conventions agree on *which* side is replaced -- otherwise
    both are right and the sets are disjoint.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm, low_is_replaced=True)

    def indicator(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)

    def mapping(x):
        v = x.copy()
        v[0] += np.isclose(x[0], 0.0) * 1.0
        v[1] += np.isclose(x[1], 0.0) * 1.0
        return v

    geometric = scifem.periodic.geometrical_search.match_vertices_geometric(
        mesh, indicator, mapping
    )
    from_gmsh = periodic_correspondence_from_nodes(mesh, pairs)

    assert_everywhere(
        comm,
        np.array_equal(
            gathered_igi(mesh, geometric.indicator_vertices),
            gathered_igi(mesh, from_gmsh.indicator_vertices),
        ),
        "the two paths replace different vertices",
    )
    assert_everywhere(
        comm,
        np.array_equal(np.sort(geometric.indicator_facets), np.sort(from_gmsh.indicator_facets)),
        "the two paths disagree on the seam facets",
    )

    # and the meshes they build agree on the invariants
    from_geometric, _, _ = scifem.periodic.mesh._build_periodic_mesh(mesh, geometric)
    from_pairs, _, _ = scifem.periodic.mesh._build_periodic_mesh(mesh, from_gmsh)
    assert torus_invariants(from_geometric)[:3] == torus_invariants(from_pairs)[:3]


def test_public_entry_point_matches_the_pieces_it_composes():
    """`create_periodic_mesh_from_igi` is the two halves, and has to stay that."""
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm)
    expected, _, _ = scifem.periodic.mesh._build_periodic_mesh(
        mesh, periodic_correspondence_from_nodes(mesh, pairs)
    )
    got, _, _ = scifem.periodic.mesh.create_periodic_mesh_from_igi(
        mesh, pairs.replaced, pairs.partner, pairs.num_nodes_global
    )
    assert torus_invariants(got)[:3] == torus_invariants(expected)[:3]


def test_read_periodic_mesh_from_msh_round_trip(tmp_path):
    """Straight from a file on disk, which is how a caller would actually use this.

    The path is chosen on rank 0 and broadcast: under MPI every process has its own
    ``tmp_path``, so letting each pick its own would have them writing and reading
    different files.
    """
    comm = MPI.COMM_WORLD
    filename = comm.bcast(str(tmp_path / "periodic.msh") if comm.rank == 0 else None, 0)

    started_here = False
    if comm.rank == 0:
        if not gmsh.isInitialized():
            gmsh.initialize()
            started_here = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("to file")
        gmsh.model.occ.addRectangle(0, 0, 0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setPeriodic(1, [2], [4], [1, 0, 0, 1.0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        gmsh.model.mesh.setPeriodic(1, [3], [1], [1, 0, 0, 0, 0, 1, 0, 1.0, 0, 0, 1, 0, 0, 0, 0, 1])
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0 / 5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0 / 5)
        gmsh.model.mesh.generate(2)
        gmsh.write(filename)
        if started_here:
            gmsh.finalize()
    comm.Barrier()

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
        kwargs = {"partitioner": dolfinx.mesh.create_cell_partitioner(ghost_mode, **part_kwargs)}
    periodic_mesh, _, _ = read_periodic_mesh_from_msh(filename, comm, gdim=2, **kwargs)

    num_vertices, volume, bad, jump = torus_invariants(periodic_mesh)
    assert (num_vertices, bad) == (33, 0)
    assert np.isclose(volume, 1.0)
    assert jump < 1e-12


def periodic_box(comm, res=1.0 / 4, low_is_replaced=True, order=1):
    """A distributed unit cube from gmsh, periodic in all three directions.

    Surface tags of ``occ.addBox`` are 1 at x=0, 2 at x=1, 3 at y=0, 4 at y=1, 5 at z=0
    and 6 at z=1 -- checked against the bounding boxes rather than assumed.

    The default resolution is not arbitrary. At ``1/3`` the glued mesh is not a manifold:
    206 cells, and 11 owned facets end up with other than two of them, the same way a 2x2
    doubly periodic square fails because each cell meets its neighbour on both sides. The
    pairing is still correct there -- the seam jump is 2e-15 -- so this is a property of
    the mesh, not of the identification. From ``1/4`` on it is clean.

    Returns:
        ``(mesh, pairs)``, with `pairs` meaningful on rank 0 only.
    """
    started_here = False
    if comm.rank == 0:
        if not gmsh.isInitialized():
            gmsh.initialize()
            started_here = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("periodic box")
        gmsh.model.occ.addBox(0, 0, 0, 1.0, 1.0, 1.0)
        gmsh.model.occ.synchronize()
        low, high = (1, 3, 5), (2, 4, 6)
        for direction in range(3):
            shift = [0.0, 0.0, 0.0]
            # gmsh stores the transform as partner -> replaced, so it points from the side
            # that survives towards the side that is replaced.
            shift[direction] = -1.0 if low_is_replaced else 1.0
            replaced, partner = (
                ([low[direction]], [high[direction]])
                if low_is_replaced
                else ([high[direction]], [low[direction]])
            )
            gmsh.model.mesh.setPeriodic(
                2,
                replaced,
                partner,
                [
                    1,
                    0,
                    0,
                    shift[0],
                    0,
                    1,
                    0,
                    shift[1],
                    0,
                    0,
                    1,
                    shift[2],
                    0,
                    0,
                    0,
                    1,
                ],
            )
        gmsh.model.addPhysicalGroup(3, [1], 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", res)
        gmsh.option.setNumber("Mesh.MeshSizeMax", res)
        gmsh.model.mesh.generate(3)
        if order > 1:
            gmsh.model.mesh.setOrder(order)
        pairs = extract_gmsh_periodic_nodes(gmsh.model)
    else:
        pairs = PeriodicNodes()

    mesh = _model_to_mesh(comm, 0, gdim=3)
    if started_here:
        gmsh.finalize()
    return mesh, pairs


def test_gmsh_path_in_3d_builds_a_three_torus():
    """A triply periodic cube, where the corner chain runs through three directions.

    2D only ever exercises a two-hop chain and facets that are edges; here the corner at
    the origin has to reach (1,1,1), and the seam facets are triangles.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_box(comm)
    correspondence = periodic_correspondence_from_nodes(mesh, pairs)
    periodic_mesh, _, _ = scifem.periodic.mesh._build_periodic_mesh(mesh, correspondence)

    num_vertices, volume, bad, jump = torus_invariants(periodic_mesh)
    assert bad == 0, f"{bad} owned facet(s) do not carry two cells"
    assert np.isclose(volume, 1.0)
    assert jump < 1e-12, f"seam jumps by {jump:.3e}"

    before = mesh.topology.index_map(0).size_global
    num_replaced = comm.bcast(len(pairs.replaced) if comm.rank == 0 else None, root=0)
    assert num_vertices == before - num_replaced


def test_gmsh_and_geometric_paths_agree_in_3d():
    """The equivalence test in 3D, which is where the corner chain is longest.

    The `mapping` applies every offset that applies to the point, so it reaches the root in
    one call; gmsh records the same identification as three separate surface pairings that
    the reader has to compose. That the two agree is the substance of this test.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_box(comm, low_is_replaced=True)

    def indicator(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[2], 0.0)

    def mapping(x):
        v = x.copy()
        for d in range(3):
            v[d] += np.isclose(x[d], 0.0) * 1.0
        return v

    geometric = scifem.periodic.geometrical_search.match_vertices_geometric(
        mesh, indicator, mapping
    )
    from_gmsh = periodic_correspondence_from_nodes(mesh, pairs)

    assert_everywhere(
        comm,
        np.array_equal(
            gathered_igi(mesh, geometric.indicator_vertices),
            gathered_igi(mesh, from_gmsh.indicator_vertices),
        ),
        "the two paths replace different vertices",
    )
    assert_everywhere(
        comm,
        np.array_equal(np.sort(geometric.indicator_facets), np.sort(from_gmsh.indicator_facets)),
        "the two paths disagree on the seam facets",
    )

    from_geometric, _, _ = scifem.periodic.mesh._build_periodic_mesh(mesh, geometric)
    from_pairs, _, _ = scifem.periodic.mesh._build_periodic_mesh(mesh, from_gmsh)
    assert torus_invariants(from_geometric)[:3] == torus_invariants(from_pairs)[:3]


@pytest.mark.parametrize("order", [1, 2])
def test_gmsh_path_on_a_second_order_mesh(order):
    """A P2 gmsh mesh through the reader, checked by the same torus invariants as P1.

    The two orders are run side by side so the P2 result is read against the P1 one rather
    than against a number written down here. What the rebuild does with the extra nodes is
    nothing: they are geometry, and geometry is never merged, so the node count comes out
    the same as it went in.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm, order=order)
    num_nodes = mesh.geometry.index_map().size_global
    num_vertices = mesh.topology.index_map(0).size_global
    if order == 1:
        assert num_nodes == num_vertices, "a P1 mesh's nodes are exactly its vertices"
    else:
        assert num_nodes > num_vertices, f"P{order} added no nodes beyond the vertices"

    periodic = scifem.periodic.mesh.create_periodic_mesh_from_igi(
        mesh, pairs.replaced, pairs.partner, pairs.num_nodes_global
    )[0]
    _, volume, bad, jump = torus_invariants(periodic)

    assert bad == 0, f"{bad} owned facets do not have two cells"
    assert np.isclose(volume, 1.0)
    assert jump < 1e-10, f"a periodic field jumps by {jump:.3e} across the seam"
    # geometry nodes are never merged, so the node count is untouched by the rebuild
    assert periodic.geometry.index_map().size_global == num_nodes


def test_raising_the_mesh_order_does_not_change_which_vertices_are_replaced():
    """The same physical vertices are replaced whatever the geometry order.

    The reader pairs nodes, and a P2 model has nodes that are not vertices; this is the
    end-to-end statement that they do not reach the correspondence. The serial test above
    is the one that contrasts the two settings of `include_high_order`; this one fixes the
    setting and varies the mesh instead.

    Compared by coordinate rather than by input global index, because `setOrder(2)` inserts
    the midside nodes into gmsh's numbering and so renumbers the vertices -- the same
    physical vertex has a different tag in the two models. Gathering and sorting keeps the
    comparison independent of the partition.
    """
    comm = MPI.COMM_WORLD
    replaced_at = {}
    node_count = {}
    for order in (1, 2):
        mesh, pairs = periodic_square(comm, order=order)
        _, replaced_vertices, _ = scifem.periodic.mesh.create_periodic_mesh_from_igi(
            mesh, pairs.replaced, pairs.partner, pairs.num_nodes_global
        )
        owned = replaced_vertices[replaced_vertices < mesh.topology.index_map(0).size_local]
        nodes = dolfinx.mesh.entities_to_geometry(mesh, 0, owned).reshape(-1)
        x = np.vstack(comm.allgather(mesh.geometry.x[nodes, : mesh.geometry.dim]))
        replaced_at[order] = x[np.lexsort(x.T)]
        node_count[order] = mesh.geometry.index_map().size_global

    assert node_count[2] > node_count[1], "the P2 mesh carries no extra nodes"
    assert len(replaced_at[1]) == len(replaced_at[2])
    assert np.allclose(replaced_at[1], replaced_at[2]), (
        "raising the geometry order changed which vertices are replaced"
    )


def test_the_empty_correspondence_needs_no_arguments():
    """The ranks that hold nothing say so by saying nothing.

    Every process other than the reader passes an empty set, so that is the default rather
    than something each caller assembles.
    """
    pairs = PeriodicNodes()
    assert len(pairs.replaced) == 0 and len(pairs.partner) == 0
    assert pairs.replaced.dtype == np.int64 and pairs.partner.dtype == np.int64
    assert pairs.num_nodes_global == 0
    # a default_factory, not a shared array: two instances must not alias
    assert PeriodicNodes().replaced is not pairs.replaced


def test_a_node_count_too_small_for_the_pairs_is_rejected():
    """`num_nodes_global` keys every post office, and getting it wrong misroutes silently.

    It is the one field that cannot be derived from the mesh, and the one a caller who took
    the default would leave at zero, so it is checked against the tags themselves.
    """
    comm = MPI.COMM_WORLD
    mesh, pairs = periodic_square(comm)
    largest = comm.bcast(
        int(max(pairs.replaced.max(), pairs.partner.max())) if comm.rank == 0 else None,
        0,
    )
    too_small = PeriodicNodes(
        pairs.replaced,
        pairs.partner,
        largest,  # one short: tags are 0-based
    )
    with pytest.raises(RuntimeError, match="num_nodes_global"):
        periodic_correspondence_from_nodes(mesh, too_small)
