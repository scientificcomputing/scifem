# Tests for create_periodic_mesh
# SPDX-License-Identifier: MIT

"""Periodicity is checked by assembling across the seam, not by counting.

A `mapping_function` that is wrong by a whole cell produces a mesh with exactly the
right vertex count, the right cell count and the right volume. The only thing that
separates it from a correct one is the jump of a periodic field across the facets that
the merge turned into interior facets.

Run serially, or under MPI::

    python3 -m pytest test_periodic.py
    mpirun -n 3 python3 -m pytest test_periodic.py
"""

from mpi4py import MPI
import inspect

import basix.ufl
import numpy as np
import pytest
import ufl

import dolfinx

from scifem.compat import create_partitioner
from scifem.periodic.geometrical_search import match_vertices_geometric
from scifem.periodic.mesh import create_periodic_mesh
from scifem.periodic.mesh import (
    DEFAULT_TAG_BASE,
    check_cells_stayed_distinct,
    VertexCorrespondence,
    _build_periodic_mesh,
    NUM_CONSENSUS_TAGS,
    check_facet_ghosting,
)


def unit_square(n=8, offset=0.0):
    """A unit square, optionally translated away from the origin."""
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    if offset != 0.0:
        mesh.geometry.x[:, :2] += offset
    return mesh


def seam_jump(periodic_mesh, field):
    """sqrt of the integral of the squared jump of `field` over all interior facets.

    For a DG-1 interpolant of a smooth function this is zero on every ordinary interior
    facet, so anything above roundoff comes from the merged seam.
    """
    V = dolfinx.fem.functionspace(periodic_mesh, ("DG", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(field)
    form = dolfinx.fem.form(ufl.jump(u) ** 2 * ufl.dS)
    local = dolfinx.fem.assemble_scalar(form)
    return np.sqrt(periodic_mesh.comm.allreduce(local, op=MPI.SUM))


def volume(mesh):
    form = dolfinx.fem.form(1 * ufl.dx(domain=mesh))
    return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


# --------------------------------------------------------------------------- #
# x-periodicity
# --------------------------------------------------------------------------- #


def x_periodic(offset=0.0, seam_shift=0.0, n=8):
    """indicator/mapping for x-periodicity.

    `seam_shift` slides the partner selection along the seam by that many cells,
    wrapped so every mapped point stays strictly inside the domain. Any nonzero
    value is a wrong mapping that still lands exactly on a real vertex.
    """
    lo, hi = offset, offset + 1.0
    h = 1.0 / n

    def indicator(x):
        # absolute tolerance: np.isclose is relative by default, and at x ~ 1e6 its
        # default rtol * |b| is 10.0, which would flag the whole domain as the seam
        return np.isclose(x[0], lo, rtol=0.0, atol=1e-4 * h)

    def mapping(x):
        v = x.copy()
        on = indicator(x)
        v[0] = np.where(on, hi, v[0])
        if seam_shift:
            shifted = lo + np.mod(v[1] - lo + seam_shift * h, 1.0)
            v[1] = np.where(on, shifted, v[1])
        return v

    return indicator, mapping


@pytest.mark.parametrize("n", [4, 8])
def test_x_periodic_mesh_is_periodic(n):
    """The assembly check: a periodic field has no jump across the merged seam."""
    mesh = unit_square(n)
    indicator, mapping = x_periodic(n=n)
    pm, _, _ = create_periodic_mesh(mesh, indicator, mapping)

    # one column of vertices is merged away
    assert pm.topology.index_map(0).size_global == (n + 1) * n
    assert pm.topology.index_map(pm.topology.dim).size_global == 2 * n * n
    assert np.isclose(volume(pm), 1.0)

    jump = seam_jump(pm, lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    assert jump < 1e-12, f"field is periodic but jumps by {jump:.3e} across the seam"


@pytest.mark.parametrize("seam_shift", [0.5, 0.25, 1.5])
def test_mapping_that_misses_a_vertex_is_rejected(seam_shift):
    """A mapped point that lands between vertices is caught by the snap distance.

    Before the distance was checked this built a mesh with the same vertex count, the
    same cell count and the same volume as the correct one, and a seam jump of O(1).
    """
    n = 8
    mesh = unit_square(n)
    indicator, mapping = x_periodic(seam_shift=seam_shift, n=n)
    with pytest.raises(RuntimeError, match="did not map"):
        create_periodic_mesh(mesh, indicator, mapping)


@pytest.mark.parametrize("seam_shift", [1.0, 2.0])
def test_lattice_aligned_shift_is_a_different_gluing(seam_shift):
    """The limit of specifying periodicity geometrically -- not a defect to be fixed.

    Sliding the partner selection by a whole number of cells still lands exactly on a real
    vertex, so the snap distance is zero and no local check can object. Nor should it:
    the seam is mapped onto itself, so the result is a genuine periodic mesh, glued with
    a twist. It is wrong only relative to what the caller meant, and `create_periodic_mesh`
    cannot see intent.

    The consequence is that counts prove nothing. This mesh has the same vertex count,
    cell count and volume as the correct one; only assembling a field that is periodic
    under the *intended* identification separates them.
    """
    n = 8
    mesh = unit_square(n)
    indicator, mapping = x_periodic(seam_shift=seam_shift, n=n)
    pm, _, _ = create_periodic_mesh(mesh, indicator, mapping)

    # indistinguishable from the correct mesh by counting
    assert pm.topology.index_map(0).size_global == (n + 1) * n
    assert pm.topology.index_map(pm.topology.dim).size_global == 2 * n * n
    assert np.isclose(volume(pm), 1.0)

    # and yet not periodic in the direction that was asked for
    jump = seam_jump(pm, lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    assert jump > 1e-2, f"expected a twisted gluing to show a jump, got {jump:.3e}"


def test_domain_far_from_origin():
    """The snap tolerance is relative, so a translated domain must still pass.

    At coordinates around 1e6 a single representable double is ~1.2e-10, already larger
    than `10000 * finfo.eps` = 2.2e-12. Compared as an absolute length the check would
    reject this correct mesh.
    """
    n = 8
    offset = 1e6
    mesh = unit_square(n, offset=offset)
    indicator, mapping = x_periodic(offset=offset, n=n)
    pm, _, _ = create_periodic_mesh(mesh, indicator, mapping)

    assert pm.topology.index_map(0).size_global == (n + 1) * n
    assert np.isclose(volume(pm), 1.0)
    jump = seam_jump(pm, lambda x: np.cos(2 * np.pi * (x[0] - offset)) * np.cos(2 * np.pi * x[1]))
    # the floor here is coordinate roundoff, not the algorithm: one representable double
    # at 1e6 is ~1.2e-10, so the interpolated field carries ~1e-9 of noise. A wrong
    # pairing still shows up at O(0.1), so the check keeps all its discriminating power.
    assert jump < 1e-6, f"translated domain jumps by {jump:.3e} across the seam"


# --------------------------------------------------------------------------- #
# two-direction periodicity
# --------------------------------------------------------------------------- #


def test_doubly_periodic_mesh_is_periodic():
    """Both offsets composed on the corner. The corner's partner is then the opposite
    corner, which is not itself a replaced."""
    n = 8
    mesh = unit_square(n)

    def i_x(x):
        return np.isclose(x[0], 0.0)

    def i_y(x):
        return np.isclose(x[1], 0.0)

    def indicator(x):
        return i_x(x) | i_y(x)

    def mapping(x):
        v = x.copy()
        v[0] += i_x(x) * 1.0
        v[1] += i_y(x) * 1.0
        return v

    pm, _, _ = create_periodic_mesh(mesh, indicator, mapping)

    # a torus: one row and one column merged away, the corner counted once
    assert pm.topology.index_map(0).size_global == n * n
    assert np.isclose(volume(pm), 1.0)
    jump = seam_jump(pm, lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    assert jump < 1e-12, f"doubly periodic mesh jumps by {jump:.3e}"


def test_per_direction_corner_mapping_resolves_to_the_same_mesh():
    """Each offset applied on its own, which sends the corner onto another replaced.

    The corner (0,0) maps to (1,0), which `indicator` also selects, so its partner has
    already been removed from the reduced index map. `create_periodic_mesh` follows the
    mapping again, reaching (1,1), and the result is the same torus the composed mapping
    gives. Corner handling is not the caller's problem.
    """
    n = 8
    mesh = unit_square(n)

    def i_x(x):
        return np.isclose(x[0], 0.0)

    def i_y(x):
        return np.isclose(x[1], 0.0)

    def indicator(x):
        return i_x(x) | i_y(x)

    def mapping(x):
        v = x.copy()
        on_x = i_x(x)
        v[0] += on_x * 1.0
        v[1] += (~on_x & i_y(x)) * 1.0
        return v

    pm, _, _ = create_periodic_mesh(mesh, indicator, mapping)

    assert pm.topology.index_map(0).size_global == n * n
    assert np.isclose(volume(pm), 1.0)
    jump = seam_jump(pm, lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    assert jump < 1e-12, f"chain-resolved mapping jumps by {jump:.3e}"


def test_cyclic_mapping_is_rejected():
    """A mapping that never leaves `indicator` has to stop, not spin."""
    n = 4
    mesh = unit_square(n)

    def indicator(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)

    def mapping(x):
        # swaps the two sides forever: neither image is ever outside `indicator`
        v = x.copy()
        v[0] = np.where(np.isclose(x[0], 0.0), 1.0, 0.0)
        return v

    with pytest.raises(RuntimeError, match="did not reach a vertex outside"):
        create_periodic_mesh(mesh, indicator, mapping)


def test_quadrilateral_corner_cell():
    """A quad corner cell carries an indicator facet on x=0 *and* on y=0.

    The cell-per-facet packing has to stay indexed by facet: there are more indicator
    facets than distinct incident cells, so collapsing to the unique set of cells breaks
    the alignment.
    """
    # enough cells that every rank keeps a shared-facet halo: the rebuilt mesh does not
    # guarantee one for an interprocess facet when a rank owns only a couple of cells
    n = 8
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        n,
        n,
        cell_type=dolfinx.mesh.CellType.quadrilateral,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )

    def i_x(x):
        return np.isclose(x[0], 0.0)

    def i_y(x):
        return np.isclose(x[1], 0.0)

    def indicator(x):
        return i_x(x) | i_y(x)

    def mapping(x):
        v = x.copy()
        v[0] += i_x(x) * 1.0
        v[1] += i_y(x) * 1.0
        return v

    pm, _, _ = create_periodic_mesh(mesh, indicator, mapping)
    assert pm.topology.index_map(0).size_global == n * n
    assert np.isclose(volume(pm), 1.0)
    jump = seam_jump(pm, lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    assert jump < 1e-12, f"quad torus jumps by {jump:.3e}"


def test_quadrilateral_4x4_doubly_periodic():
    """Regression test for the ghost cell that used to go missing at 5 ranks.

    `create_periodic_mesh` shipped, from the process taking over a replacement vertex, only
    the cells behind the boundary facets that process *owned* -- `exterior_facet_indices`
    returns owned facets only. Where the partition left it merely ghosting one of those
    facets, the cell behind it was never shipped, and the process on the other side of the
    seam ended up with a seam facet holding one cell instead of two. On a doubly periodic
    mesh there is no boundary at all, so every facet must have exactly two.

    It bit at 5 ranks and not at 4 or 6, purely by how the partition lined up. The failure
    was also rank-local -- one process could not build the form while the rest ran on -- so
    the reduction below is what keeps a regression a clean failure rather than a deadlock.

    There were two causes, one per direction. Phase 1 shipped only the cells behind the
    boundary facets the process *owned*, because `exterior_facet_indices` is owned-only.
    Phase 3 sent the cell to the single owner `determine_point_ownership` returned for the
    mapped point -- but that point lands exactly on a vertex, which several cells share, so
    the other owners of that vertex's cells were left short. Both are fixed; this test fails
    at 5 ranks if either regresses, on either DOLFINx version.
    """
    comm = MPI.COMM_WORLD
    n = 4
    mesh = dolfinx.mesh.create_unit_square(
        comm,
        n,
        n,
        cell_type=dolfinx.mesh.CellType.quadrilateral,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )

    def i_x(x):
        return np.isclose(x[0], 0.0)

    def i_y(x):
        return np.isclose(x[1], 0.0)

    def indicator(x):
        return i_x(x) | i_y(x)

    def mapping(x):
        v = x.copy()
        v[0] += i_x(x) * 1.0
        v[1] += i_y(x) * 1.0
        return v

    # the mesh itself builds everywhere, on every rank count
    pm, _, _ = create_periodic_mesh(mesh, indicator, mapping)
    assert pm.topology.index_map(0).size_global == n * n
    assert np.isclose(volume(pm), 1.0)

    V = dolfinx.fem.functionspace(pm, ("DG", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))

    # `dolfinx.fem.form` is where it goes wrong, and only on some ranks. Reduce before
    # raising: assembling below is collective, so a rank that bailed out early would leave
    # the others blocked in the allreduce.
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
    assert jump < 1e-12, f"4x4 quad torus jumps by {jump:.3e}"


# --------------------------------------------------------------------------- #
# chain resolution: how many times the mapping may be re-applied
# --------------------------------------------------------------------------- #


def periodic_in(directions, *, per_direction):
    """An indicator/mapping pair identifying x_d = 0 with x_d = 1 for each d.

    With ``per_direction=False`` the mapping applies every offset that applies to the
    point, so a corner reaches its root in one application. With ``per_direction=True`` it
    applies only the first, so a corner needs one application per direction -- the case
    that fixes how large `max_chain_length` has to be.
    """

    def indicator(x):
        marked = np.zeros(x.shape[1], dtype=np.bool_)
        for d in directions:
            marked |= np.isclose(x[d], 0.0)
        return marked

    def mapping(x):
        v = x.copy()
        moved = np.zeros(x.shape[1], dtype=np.bool_)
        for d in directions:
            on_face = np.isclose(x[d], 0.0)
            if per_direction:
                on_face = on_face & ~moved
                moved |= on_face
            v[d] += on_face * 1.0
        return v

    return indicator, mapping


def unit_cube(n=3):
    return dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, n, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )


CHAIN_CASES = [
    # (name, mesh factory, directions, per_direction, applications needed)
    ("2d-single", unit_square, [0], False, 1),
    ("2d-double-composed", unit_square, [0, 1], False, 1),
    ("2d-double-per-direction", unit_square, [0, 1], True, 2),
    ("3d-single", unit_cube, [0], False, 1),
    ("3d-triple-composed", unit_cube, [0, 1, 2], False, 1),
    ("3d-triple-per-direction", unit_cube, [0, 1, 2], True, 3),
]


@pytest.mark.parametrize(
    "name,make_mesh,directions,per_direction,needed",
    CHAIN_CASES,
    ids=[c[0] for c in CHAIN_CASES],
)
def test_chain_length_bound_is_the_topological_dimension(
    name, make_mesh, directions, per_direction, needed
):
    """`max_chain_length` defaults to `tdim`, which has to be exactly the bound.

    A mapping applying one offset per call needs one application per periodic direction,
    and a tdim-manifold admits at most tdim independent ones. This pins both halves: the
    default must suffice for every legitimate mapping, and it must not be slack, or a
    mapping that never terminates would be allowed extra rounds before being caught.

    It is the *topological* dimension, not the geometric one -- a flat torus in R^3 has
    gdim 3 and only two directions to be periodic in.
    """
    mesh = make_mesh()
    indicator, mapping = periodic_in(directions, per_direction=per_direction)
    tdim = mesh.topology.dim

    assert needed <= tdim, f"{name} needs more applications than tdim allows"

    # the default resolves the chain
    match_vertices_geometric(mesh, indicator, mapping)

    # and so does exactly the number of applications this case needs
    match_vertices_geometric(mesh, indicator, mapping, max_chain_length=needed)

    # one fewer does not: the bound is real, not decorative
    if needed > 1:
        with pytest.raises(RuntimeError, match="did not reach a vertex outside"):
            match_vertices_geometric(mesh, indicator, mapping, max_chain_length=needed - 1)


def test_per_direction_corner_needs_one_application_per_direction():
    """The worst legitimate case saturates the default exactly.

    Together with the test above this is what licenses `tdim` as the default: the
    per-direction mapping in the top dimension needs all of it and no more.
    """
    for make_mesh, directions in ((unit_square, [0, 1]), (unit_cube, [0, 1, 2])):
        mesh = make_mesh()
        indicator, mapping = periodic_in(directions, per_direction=True)
        tdim = mesh.topology.dim
        match_vertices_geometric(mesh, indicator, mapping, max_chain_length=tdim)
        with pytest.raises(RuntimeError, match="did not reach a vertex outside"):
            match_vertices_geometric(mesh, indicator, mapping, max_chain_length=tdim - 1)


# --------------------------------------------------------------------------- #
# preconditions and knobs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "ghost_mode",
    [dolfinx.mesh.GhostMode.none, dolfinx.mesh.GhostMode.shared_facet],
)
def test_check_facet_ghosting_sees_the_ghost_mode(ghost_mode):
    """The check is exactly a ghost-mode test, and says nothing in serial.

    `interprocess_facets` is the same set under both modes -- that is what makes it usable
    here -- so what separates them is only how many cells each of those facets carries.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6, ghost_mode=ghost_mode)
    unghosted = MPI.COMM_WORLD.size > 1 and ghost_mode is dolfinx.mesh.GhostMode.none
    if unghosted:
        with pytest.raises(RuntimeError, match="do not have both of their cells"):
            check_facet_ghosting(mesh)
    else:
        check_facet_ghosting(mesh)


def test_unghosted_input_is_rejected_before_the_rebuild():
    """An unghosted mesh used to fail much later, as a seam facet with one cell.

    Without the precondition the diagnosis points at the seam, which is the one part of
    the mesh that is not at fault.
    """
    if MPI.COMM_WORLD.size == 1:
        pytest.skip("a serial mesh has no interprocess facets to be unghosted")
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 6, 6, ghost_mode=dolfinx.mesh.GhostMode.none
    )
    indicator, mapping = x_periodic(n=6)
    with pytest.raises(RuntimeError, match="ghost_mode"):
        create_periodic_mesh(mesh, indicator, mapping)


@pytest.mark.parametrize("tag_base", [DEFAULT_TAG_BASE, 3000])
def test_tag_base_does_not_change_the_mesh(tag_base):
    """The tags name consensus exchanges; moving them is invisible in the result."""
    n = 6
    mesh = unit_square(n)
    indicator, mapping = x_periodic(n=n)
    pm, replaced, _ = create_periodic_mesh(mesh, indicator, mapping, tag_base=tag_base)

    assert pm.topology.index_map(0).size_global == (n + 1) * n
    assert np.isclose(volume(pm), 1.0)
    assert MPI.COMM_WORLD.allreduce(len(replaced), op=MPI.SUM) > 0
    jump = seam_jump(pm, lambda x: np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    assert jump < 1e-12


def test_consecutive_tag_bases_do_not_overlap():
    """Two calls one full block apart, both collective, neither crossing the other.

    `NUM_CONSENSUS_TAGS` is the spacing a caller has to respect, so it is worth asserting
    that it really is the number of tags a rebuild consumes rather than a stale constant.
    """
    n = 6
    mesh = unit_square(n)
    indicator, mapping = x_periodic(n=n)
    first, _, _ = create_periodic_mesh(mesh, indicator, mapping, tag_base=4000)
    second, _, _ = create_periodic_mesh(
        mesh, indicator, mapping, tag_base=4000 + NUM_CONSENSUS_TAGS
    )
    assert first.topology.index_map(0).size_global == second.topology.index_map(0).size_global
    assert np.isclose(volume(first), volume(second))


def test_a_correspondence_that_misses_a_ghost_copy_is_rejected():
    """Every process holding a replaced vertex has to name it, and saying so is cheap.

    `_reduced_vertex_map` drops the *broadcast* set, so a correspondence that names a
    vertex on its owner but not on a rank that merely ghosts it leaves that rank with a
    `-1` in `replacement_map` -- which would go straight into the cell dofmap. Before the
    check this did not fail, it **hung**, and it hung *before* reaching the replacement
    map: the neighbourhood exchanges are built from the correspondence, so the ranks that
    agree sit waiting for the one that does not. The check therefore lives in
    `_reduced_vertex_map`, ahead of every exchange.

    The geometric and gmsh paths both broadcast, so this can only be reached by a caller
    building a correspondence by hand. That is exactly who the message is for.
    """
    if MPI.COMM_WORLD.size == 1:
        pytest.skip("a serial mesh has no ghost copies to leave out")
    comm = MPI.COMM_WORLD
    n = 6
    mesh = unit_square(n)
    indicator, mapping = x_periodic(n=n)
    correspondence = match_vertices_geometric(mesh, indicator, mapping)

    # Drop the ghost indicator vertices on rank 0: a caller that forgot to broadcast.
    vertex_map = mesh.topology.index_map(0)
    keep = np.ones(len(correspondence.indicator_vertices), dtype=np.bool_)
    if comm.rank == 0:
        keep = correspondence.indicator_vertices < vertex_map.size_local
    if comm.allreduce(int((~keep).sum()), op=MPI.SUM) == 0:
        pytest.skip("this partition gives rank 0 no ghost indicator vertices to drop")

    incomplete = VertexCorrespondence(
        indicator_vertices=correspondence.indicator_vertices[keep],
        indicator_facets=correspondence.indicator_facets,
        src_owner=correspondence.src_owner[keep],
        dest_owner=correspondence.dest_owner,
        partner_vertex=correspondence.partner_vertex,
    )
    with pytest.raises(RuntimeError, match="not on every process that holds them"):
        _build_periodic_mesh(mesh, incomplete)


@pytest.mark.parametrize(
    "cell_type",
    [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral],
)
def test_two_cells_across_a_seam_is_rejected(cell_type):
    """N=2 collapses, and every count that is easy to check says it did not.

    With two cells between the two sides of a seam there are only two vertices on the
    circle that direction becomes, so both cells run between the same pair and end up
    carrying the *same vertex set*. Measured before the guard: 4 vertices, 4 quadrilateral
    cells, and all 4 facets reporting 4 incident cells -- with no degenerate cell, the right
    vertex and cell counts, and volume 1.0. Only the facet-to-cell count shows it.
    """
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        2,
        2,
        cell_type=cell_type,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )
    indicator, mapping = periodic_in((0, 1), per_direction=False)
    with pytest.raises(RuntimeError, match="collapsed cells onto each other"):
        create_periodic_mesh(mesh, indicator, mapping)


def test_three_cells_across_a_seam_is_the_minimum():
    """The guard is not just refusing small meshes: one more cell and it is fine."""
    mesh = unit_square(3)
    indicator, mapping = periodic_in((0, 1), per_direction=False)
    periodic = create_periodic_mesh(mesh, indicator, mapping)[0]

    assert periodic.topology.index_map(0).size_global == 9
    assert np.isclose(volume(periodic), 1.0)
    check_cells_stayed_distinct(periodic)


def test_a_facet_with_three_distinct_cells_is_not_a_collapse():
    """The guard must not fire on geometry that is non-manifold by design.

    Three sheets meeting along an edge -- the stem of a T -- gives that edge three cells,
    and none of them is the same cell. Only a repeated vertex set is the failure, so the
    check is written on that and not on the incident-cell count, which cannot tell the two
    apart. Built here directly, since `create_periodic_mesh` is not what produces it.

    Worth running distributed as well as serially: the check reads the cells at an owned
    facet, and whether all three are there is a question about ghosting, which serial
    cannot ask. `shared_facet` is what brings the third one over.
    """
    comm = MPI.COMM_WORLD
    # `max_facet_to_cell_links` defaults to 2, which is DOLFINx refusing exactly this mesh
    # unless asked. Probed by signature rather than by version, as elsewhere in the suite.
    if "max_facet_to_cell_links" not in inspect.signature(dolfinx.mesh.create_mesh).parameters:
        pytest.skip("this DOLFINx cannot be asked for more than two cells per facet")

    # The input goes in on one rank and is distributed from there; handing every rank the
    # same cells would build the mesh `comm.size` times over.
    if comm.rank == 0:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # the shared edge
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],  # sheet in +y
                [0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],  # sheet in -y
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],  # sheet in +z, the stem
            ]
        )
        cells = np.array(
            [[0, 1, 2], [1, 3, 2], [0, 1, 4], [1, 5, 4], [0, 1, 6], [1, 7, 6]],
            dtype=np.int64,
        )
    else:
        points = np.zeros((0, 3), dtype=np.float64)
        cells = np.zeros((0, 3), dtype=np.int64)

    # Ghosting is a `create_mesh` keyword on 0.12 and a partitioner on 0.11, the same split
    # `model_to_mesh` carries in test_gmsh_periodic.py. Both places have to be told that a
    # facet may carry three cells, or the third sheet is not ghosted and the check sees two.
    partitioner = create_partitioner(dolfinx.mesh.GhostMode.shared_facet, max_facet_to_cell_links=3)
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet
    kwargs = {}
    if "ghost_mode" in inspect.signature(dolfinx.mesh.create_mesh).parameters:
        kwargs["ghost_mode"] = ghost_mode
    kwargs["partitioner"] = partitioner

    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(3,)))
    mesh = dolfinx.mesh.create_mesh(
        comm, cells, domain, points, max_facet_to_cell_links=3, **kwargs
    )
    assert mesh.topology.index_map(2).size_global == 6

    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(1, 2)
    f_to_c = mesh.topology.connectivity(1, 2)
    num_owned = mesh.topology.index_map(1).size_local
    per_facet = np.diff(f_to_c.offsets)[:num_owned]
    # Reduced: whichever rank owns the shared edge sees the 3, and the others see nothing
    # of it. `initial=0` because a rank can own no facet at all at these cell counts.
    busiest = comm.allreduce(int(per_facet.max(initial=0)), op=MPI.MAX)
    assert busiest == 3, f"the shared edge should carry all three sheets, saw {busiest}"

    check_cells_stayed_distinct(mesh)


def test_a_remaining_boundary_is_not_non_manifold():
    """Facets with one cell are a boundary, not a collapse, and must not trip the guard.

    A mesh made periodic in x only still has its y=0 and y=1 edges.
    """
    mesh = unit_square(6)
    periodic = create_periodic_mesh(mesh, *x_periodic(n=6))[0]
    check_cells_stayed_distinct(periodic)

    tdim = periodic.topology.dim
    periodic.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = periodic.topology.connectivity(tdim - 1, tdim)
    num_owned = periodic.topology.index_map(tdim - 1).size_local
    per_facet = np.diff(f_to_c.offsets)[:num_owned]
    with_one = MPI.COMM_WORLD.allreduce(int(np.count_nonzero(per_facet == 1)), op=MPI.SUM)
    assert with_one == 12, "the two non-periodic edges should still be a boundary"
