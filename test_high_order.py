# Higher-order geometry through create_periodic_mesh
# SPDX-License-Identifier: MIT

"""The merge is topological: vertices are identified, geometry nodes never are.

So a P2 or P3 input drives `num_nodes > num_vertices` through every exchange in the
rebuild, and the property that says what the algorithm does is that **the node count comes
out unchanged while the vertex count drops**. The geometry stays deliberately torn across
the seam -- the cells still sit where they sit -- which is also why the order has to be
raised *before* the merge and cannot be raised after it.

Run serially, or under MPI::

    python3 -m pytest test_high_order.py
    mpirun -n 3 python3 -m pytest test_high_order.py
"""

import basix
import dolfinx
import numpy as np
import pytest
from mpi4py import MPI

from script import create_periodic_mesh
from test_periodic import seam_jump, volume

# (cell type, geometry degree). Quadrilaterals cover the non-simplex dofmap, tetrahedra the
# 3D exchanges, and degree 3 the case with nodes interior to a facet.
CASES = [
    ("triangle", 2),
    ("quadrilateral", 2),
    ("tetrahedron", 2),
    ("triangle", 3),
]


def linear_mesh(cell_name, n):
    """A unit square or cube of `cell_name` cells, with the ghosting the rebuild needs."""
    ghost_mode = dolfinx.mesh.GhostMode.shared_facet
    if cell_name == "tetrahedron":
        return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
    return dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD,
        n,
        n,
        cell_type=getattr(dolfinx.mesh.CellType, cell_name),
        ghost_mode=ghost_mode,
    )


def higher_order(mesh, degree):
    """`mesh` with its geometry interpolated into a Lagrange element of `degree`.

    Two things about `coordinate_element` that are easy to get wrong, both encoded here:
    it takes a `dolfinx.mesh.CellType` and *not* a `basix.CellType` -- the enums have
    different integer values, so basix's is accepted and then fails as a cell-shape
    mismatch -- and degree above 2 has no default variant.
    """
    variant = 0 if degree <= 2 else int(basix.LagrangeVariant.gll_isaac)
    cmap = dolfinx.fem.coordinate_element(mesh.topology.cell_type, degree, variant)
    return dolfinx.fem.interpolate_geometry(mesh, cmap)


def discontinuous_element(cell_name, degree):
    """A Lagrange element of `degree` with nothing shared between cells.

    The last argument of `basix.create_element` is `discontinuous`; it is the `false` in
    the `create_element` call inside `interpolate_geometry`, and flipping it is what would
    let a torn geometry be represented.
    """
    return basix.create_element(
        basix.ElementFamily.P,
        getattr(basix.CellType, cell_name),
        degree,
        basix.LagrangeVariant.unset,
        basix.DPCVariant.unset,
        True,
    )


def periodic_in_every_direction(gdim):
    """Identify ``x_d = 0`` with ``x_d = 1`` in every direction at once.

    Every offset that applies to a point is applied in one call, so a corner reaches its
    root without a chain.
    """

    def indicator(x):
        marked = np.zeros(x.shape[1], dtype=np.bool_)
        for d in range(gdim):
            marked |= np.isclose(x[d], 0.0)
        return marked

    def mapping(x):
        values = x.copy()
        for d in range(gdim):
            values[d] += np.isclose(x[d], 0.0) * 1.0
        return values

    return indicator, mapping


def x_periodic():
    """Identify x=0 with x=1 and leave the other directions alone."""

    def indicator(x):
        return np.isclose(x[0], 0.0)

    def mapping(x):
        values = x.copy()
        values[0] += 1.0
        return values

    return indicator, mapping


def facets_without_two_cells(mesh):
    """Owned facets whose incident-cell count is not two, reduced over the communicator.

    Zero on a mesh with no boundary, which is what periodicity in every direction gives.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    num_owned = mesh.topology.index_map(tdim - 1).size_local
    per_facet = np.diff(f_to_c.offsets)[:num_owned]
    return mesh.comm.allreduce(int(np.count_nonzero(per_facet != 2)), op=MPI.SUM)


def geometry_degree(mesh):
    """Compat: 0.12 deprecates `geometry.cmap` in favour of `geometry.cmaps[0]`."""
    cmaps = getattr(mesh.geometry, "cmaps", None)
    return (cmaps[0] if cmaps else mesh.geometry.cmap).degree


def gathered_nodes(mesh):
    """Every owned geometry node, keyed by input global index, gathered and sorted.

    Integrating cannot see a bent interior edge at all: the edge is shared, so whatever the
    bow adds to one cell it takes from the other and the union is unchanged. The nodes
    themselves are therefore the only place the curvature actually lives, and comparing
    them is both exact and partition-independent.
    """
    num_owned = mesh.geometry.index_map().size_local
    igi = mesh.geometry.input_global_indices[:num_owned]
    x = mesh.geometry.x[:num_owned, : mesh.geometry.dim]
    all_igi = np.concatenate(mesh.comm.allgather(igi))
    all_x = np.vstack(mesh.comm.allgather(x))
    order = np.argsort(all_igi)
    return all_igi[order], all_x[order]


def periodic_field(x):
    """Periodic under a unit translation in every direction the mesh has."""
    value = np.ones(x.shape[1])
    for d in range(x.shape[0]):
        value = value * np.cos(2 * np.pi * x[d])
    return value


# --------------------------------------------------------------------------- #
# raising the order first, which is the way that works
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cell_name,degree", CASES, ids=[f"{c}-P{d}" for c, d in CASES])
def test_higher_order_geometry_survives_the_merge(cell_name, degree):
    """A curved mesh comes out periodic, not merely a mesh with the right counts.

    The jump is the half of this that counting cannot reach: a wrong pairing gives the
    same vertex count, cell count and volume as a right one.
    """
    n = 3 if cell_name == "tetrahedron" else 6
    mesh = higher_order(linear_mesh(cell_name, n), degree)
    assert geometry_degree(mesh) == degree

    indicator, mapping = periodic_in_every_direction(mesh.geometry.dim)
    periodic = create_periodic_mesh(mesh, indicator, mapping)[0]

    assert geometry_degree(periodic) == degree, "the merge lowered the geometry degree"
    assert np.isclose(volume(periodic), 1.0)
    assert facets_without_two_cells(periodic) == 0, "the merged mesh still has a boundary"
    assert seam_jump(periodic, periodic_field) < 1e-10


def test_vertices_drop_and_nodes_do_not():
    """The property that says what this algorithm is: it merges vertices, not geometry.

    A 6x6 P2 triangle mesh has 49 vertices and 169 nodes. Identifying x=0 with x=1 removes
    one column of 7 vertices and leaves every node in place, because the two sides of the
    seam keep their own coordinates -- the cells have not moved.
    """
    mesh = higher_order(linear_mesh("triangle", 6), 2)
    vertices_before = mesh.topology.index_map(0).size_global
    nodes_before = mesh.geometry.index_map().size_global
    assert (vertices_before, nodes_before) == (49, 169)

    periodic = create_periodic_mesh(mesh, *x_periodic())[0]

    assert periodic.topology.index_map(0).size_global == vertices_before - 7
    assert periodic.geometry.index_map().size_global == nodes_before


def test_curvature_is_carried_through_the_merge():
    """A genuinely curved mesh, so the test fails if the rebuild re-linearises it.

    The vertices are untouched, so every straight-sided quantity is blind to this; only the
    midside nodes move, and only an integrand that weights the two sides of a bent edge
    differently can see them.
    """
    curved = higher_order(linear_mesh("triangle", 6), 2)

    # Bow the midside nodes by a smooth field. Vertices sit on multiples of 1/6, so a
    # y-coordinate off that lattice belongs to a midside node; the vertices are left where
    # they are, so the pairing sees exactly what it would on the straight mesh.
    x = curved.geometry.x
    midside = ~np.isclose(np.round(x[:, 1] * 6) / 6, x[:, 1])
    assert midside.sum() > 0
    x[midside, 1] += 0.04 * np.sin(2 * np.pi * x[midside, 0])
    igi_before, x_before = gathered_nodes(curved)

    periodic = create_periodic_mesh(curved, *x_periodic())[0]
    igi_after, x_after = gathered_nodes(periodic)

    assert np.array_equal(igi_after, igi_before), "the owned geometry nodes changed"
    assert np.allclose(x_after, x_before), "the merge moved the geometry"


# --------------------------------------------------------------------------- #
# raising the order afterwards, which does not
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    strict=True,
    reason=(
        "`interpolate_geometry` builds its dofmap from a *continuous* space"
        " (`create_functionspace` in cpp/dolfinx/fem/utils.h:1070-1122), so the two cells"
        " meeting across the seam share one dof and push forward to two different physical"
        " points. Only one value can be stored, and the seam cells are stretched across the"
        " domain: volume 2.0 serially and 2.4-3.0 in parallel, against 1.0. The coordinates"
        " themselves are computed correctly -- the obstacle is the continuity, and it bites"
        " at any degree, including a same-degree round trip. A discontinuous element fixes"
        " it: FEniCS/dolfinx#4544, `Discontinuous CoordinateElement`, exposes"
        " `is_discontinuous` and makes `interpolate_geometry` respect it, for curving"
        " periodic meshes. Note this test stays red even then -- it passes a *continuous*"
        " element, which is still wrong. When #4544 lands, add a second test that passes a"
        " discontinuous one and expect it to pass; leave this one as the record that the"
        " default does not work."
    ),
)
def test_geometry_can_be_raised_after_the_merge():
    """Interpolating the geometry must not change the domain it describes."""
    mesh = linear_mesh("triangle", 6)
    periodic = create_periodic_mesh(mesh, *x_periodic())[0]
    assert np.isclose(volume(periodic), 1.0)

    raised = higher_order(periodic, 2)
    assert np.isclose(volume(raised), 1.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "needs FEniCS/dolfinx#4544, `Discontinuous CoordinateElement`, which exposes"
        " `is_discontinuous` and makes `interpolate_geometry` respect it -- its stated"
        " motivation is curving periodic meshes. Today the flag is accepted and ignored:"
        " a discontinuous P2 element on the 6x6 periodic mesh still yields 156 nodes, the"
        " *continuous* count, instead of 6 per cell. Building a form on the result then"
        " fails as well, because the UFL domain says discontinuous while the geometry is"
        " not. Replicated by hand, a genuinely discontinuous geometry gives volume 1.0 at"
        " degree 1 and 2 and assembles interior-facet forms, so this should pass once the"
        " PR lands. Two things to expect when it does: the node count grows a lot, since a"
        " discontinuous geometry duplicates everywhere and not only at the seam, and"
        " `input_global_indices` is regenerated, which"
        " `transfer_meshtags_to_periodic_mesh` reads."
    ),
)
def test_geometry_can_be_raised_after_the_merge_with_a_discontinuous_element():
    """The route that should work: tear the geometry, then raise its order discontinuously.

    The node count is asserted before the volume on purpose. It is the direct statement
    that the element was respected -- one set of nodes per cell, nothing shared -- and it
    is what fails today, rather than the test tripping over some later consequence.
    """
    mesh = linear_mesh("triangle", 6)
    periodic = create_periodic_mesh(mesh, *x_periodic())[0]

    element = discontinuous_element("triangle", 2)
    raised = dolfinx.fem.interpolate_geometry(periodic, dolfinx.fem.coordinate_element(element))

    num_cells = periodic.topology.index_map(periodic.topology.dim).size_global
    assert raised.geometry.index_map().size_global == num_cells * element.dim, (
        "the geometry is still shared between cells, so the seam cannot be torn"
    )
    assert np.isclose(volume(raised), 1.0)
