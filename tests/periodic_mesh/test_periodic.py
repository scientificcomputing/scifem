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
import numpy as np
import pytest
import ufl

import dolfinx

from scifem.periodic.geometrical_search import match_vertices_geometric
from scifem.periodic.mesh import create_periodic_mesh


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

    `seam_shift` slides the master selection along the seam by that many cells,
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

    Sliding the master selection by a whole number of cells still lands exactly on a real
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
    """Both offsets composed on the corner. The corner's master is then the opposite
    corner, which is not itself a slave."""
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
    """Each offset applied on its own, which sends the corner onto another slave.

    The corner (0,0) maps to (1,0), which `indicator` also selects, so its master has
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
