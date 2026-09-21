# Tests for transfer_function_to_parent_mesh
# SPDX-License-Identifier: MIT

"""The per-cell copy back onto the parent mesh is checked against interpolation there.

Correctness rests on ``cell_dofs(c)[i]`` denoting the same physical point in both meshes.
Merging the seam renumbers the vertices, which for degree 3 and above flips the order of
the two dofs on some edges, so the degrees are tested past that threshold. The test
functions must be periodic: a non-periodic one is ill-defined on the periodic mesh, and
the resulting seam garbage would swamp any ordering error.

The mesh comes from gmsh rather than ``create_unit_square``, for two reasons. Its cells
are unstructured, so the orientations the copy has to be invariant to are not laid out in
the regular pattern a structured square repeats. And its periodicity is read from the
model's ``$Periodic`` section, so the fixture exercises
{py:func}`scifem.periodic.create_periodic_mesh_from_igi` rather than the geometric
search -- the path a caller reading a ``.msh`` file takes.
"""

from mpi4py import MPI

import gmsh
import numpy as np
import pytest

import dolfinx
import inspect
from scifem.periodic.gmsh import extract_gmsh_periodic_nodes
from scifem.periodic.mesh import create_periodic_mesh_from_igi
from scifem.periodic.transfer import transfer_function_to_parent_mesh
from scifem.periodic.utils import PeriodicNodes


def _periodic_scalar(x):
    """Periodic, and asymmetric along every edge so a flipped dof pair shows up."""
    return (
        np.exp(np.sin(2 * np.pi * x[0])) * np.cos(2 * np.pi * x[1] + 0.4)
        + 0.7 * np.sin(2 * np.pi * (x[0] + x[1]) + 0.3)
        + 0.4 * np.cos(4 * np.pi * x[0] - 0.9)
    )


def _periodic_vector(x):
    return np.vstack([_periodic_scalar(x), _periodic_scalar(x[::-1])])


@pytest.fixture(scope="module")
def meshes():
    """An unstructured doubly periodic unit square, and the mesh it was built from."""
    comm = MPI.COMM_WORLD
    resolution = 0.12
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if comm.rank == 0:
            gmsh.model.add("periodic_square")
            gmsh.model.occ.addRectangle(0, 0, 0, 1.0, 1.0)
            gmsh.model.occ.synchronize()
            # `model_to_mesh` reads cells out of the physical groups, so the surface
            # needs one.
            gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], tag=1)
            # Curve tags of addRectangle: 1 bottom, 2 right, 3 top, 4 left. The right
            # curve is replaced by the left, and the top by the bottom.
            gmsh.model.mesh.setPeriodic(
                1, [2], [4], [1, 0, 0, 1.0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            )
            gmsh.model.mesh.setPeriodic(
                1, [3], [1], [1, 0, 0, 0, 0, 1, 0, 1.0, 0, 0, 1, 0, 0, 0, 0, 1]
            )
            gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
            gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)
            gmsh.model.mesh.generate(2)
            pairs = extract_gmsh_periodic_nodes(gmsh.model)
        else:
            pairs = PeriodicNodes(num_nodes_global=0)
        # The rebuild needs a ghost layer across every interprocess facet, which
        # `model_to_mesh` does not give by default.
        gmodel_to_mesh = inspect.signature(dolfinx.io.gmsh.model_to_mesh)
        kwargs = {}
        if "ghost_mode" in gmodel_to_mesh.parameters:
            kwargs["ghost_mode"] = dolfinx.mesh.GhostMode.shared_facet
        mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, comm, 0, gdim=2, **kwargs)
    finally:
        gmsh.finalize()

    mesh = getattr(mesh_data, "mesh", mesh_data)
    periodic_mesh, _, _ = create_periodic_mesh_from_igi(
        mesh, pairs.replaced, pairs.partner, pairs.num_nodes_global, root=0
    )
    return mesh, periodic_mesh


@pytest.mark.parametrize("family", ["Lagrange", "Discontinuous Lagrange"])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(), (2,)])
def test_transfer_matches_interpolation_on_the_parent(meshes, family, degree, shape):
    """The copy must agree with interpolating the same field on the parent mesh."""
    mesh, periodic_mesh = meshes
    expr = _periodic_vector if shape else _periodic_scalar

    u = dolfinx.fem.Function(dolfinx.fem.functionspace(periodic_mesh, (family, degree, shape)))
    u.interpolate(expr)

    transferred = transfer_function_to_parent_mesh(u, mesh)

    expected = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, (family, degree, shape)))
    expected.interpolate(expr)

    np.testing.assert_allclose(transferred.x.array, expected.x.array, atol=1e-13)


@pytest.mark.parametrize("degree", [3, 4, 5])
def test_transfer_survives_the_dof_reordering(meshes, degree):
    """Reversing the edge dofs on the cells whose permutation changed must break it.

    Guards the test above against being vacuous: if the per-cell copy were insensitive to
    dof ordering, the deliberately reordered copy would pass too.
    """
    mesh, periodic_mesh = meshes
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    mesh.topology.create_entity_permutations()
    periodic_mesh.topology.create_entity_permutations()
    reordered = np.flatnonzero(
        mesh.topology.get_cell_permutation_info()[:num_cells]
        != periodic_mesh.topology.get_cell_permutation_info()[:num_cells]
    )
    assert mesh.comm.allreduce(len(reordered), op=MPI.SUM) > 0, (
        "no cell changed orientation, so this test would be vacuous"
    )

    V = dolfinx.fem.functionspace(periodic_mesh, ("Lagrange", degree))
    Vp = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    u = dolfinx.fem.Function(V)
    u.interpolate(_periodic_scalar)
    edge_dofs = [Vp.dofmap.dof_layout.entity_dofs(1, e) for e in range(3)]

    corrupted = dolfinx.fem.Function(Vp)
    for cell in range(num_cells):
        source = np.asarray(V.dofmap.cell_dofs(cell)).copy()
        if cell in reordered:
            for positions in edge_dofs:
                source[positions] = source[positions][::-1]
        corrupted.x.array[Vp.dofmap.cell_dofs(cell)] = u.x.array[source]
    corrupted.x.scatter_forward()

    # Collective: a rank whose cells all kept their orientation sees no difference, so
    # the comparison has to be reduced over the communicator.
    expected = transfer_function_to_parent_mesh(u, mesh)
    local = np.abs(corrupted.x.array - expected.x.array).max(initial=0.0)
    assert mesh.comm.allreduce(local, op=MPI.MAX) > 1e-13


@pytest.mark.parametrize("family", ["RT", "N1curl", "BDM"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_transfer_handles_elements_needing_dof_transformations(meshes, family, degree):
    """RT/N1curl/BDM apply their transformations at assembly rather than folding them into
    the dofmap, so the transfer has to account for the orientations the two meshes disagree
    on. The result must still match interpolating on the parent."""
    mesh, periodic_mesh = meshes
    V = dolfinx.fem.functionspace(periodic_mesh, (family, degree))
    assert V.element.needs_dof_transformations, (
        f"{family}{degree} folds its transformations into the dofmap, so this test is vacuous"
    )
    u = dolfinx.fem.Function(V)
    u.interpolate(_periodic_vector)

    transferred = transfer_function_to_parent_mesh(u, mesh)

    expected = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, (family, degree)))
    expected.interpolate(_periodic_vector)

    np.testing.assert_allclose(transferred.x.array, expected.x.array, atol=1e-13)


def test_rejects_an_unrelated_parent_mesh(meshes):
    _, periodic_mesh = meshes
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(periodic_mesh, ("Lagrange", 1)))
    other = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    with pytest.raises(ValueError, match="same cells"):
        transfer_function_to_parent_mesh(u, other)


def test_name_is_preserved(meshes):
    mesh, periodic_mesh = meshes
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(periodic_mesh, ("Lagrange", 2)), name="uh")
    assert transfer_function_to_parent_mesh(u, mesh).name == "uh"
