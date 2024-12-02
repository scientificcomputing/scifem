from mpi4py import MPI
import numpy as np
import dolfinx
import scifem

import pytest


@pytest.mark.parametrize(
    "entities_list, set_values",
    [
        ([(1, lambda x: x[0] <= 1.0)], {1}),
        ([(2, lambda x: x[0] <= 1.0)], {2}),
        (
            [
                (1, lambda x: x[0] <= 1.0),
                (2, lambda x: x[0] >= 0.0),
            ],
            {2},
        ),
    ],
)
def test_create_celltags_celltags_all(entities_list, set_values):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    cell_tag = scifem.create_entity_markers(mesh, mesh.topology.dim, entities_list=entities_list)

    im = mesh.topology.index_map(mesh.topology.dim)
    assert cell_tag.dim == mesh.topology.dim
    assert cell_tag.indices.shape[0] == im.size_local + im.num_ghosts
    assert cell_tag.values.shape[0] == im.size_local + im.num_ghosts
    assert cell_tag.values.dtype == np.int32
    assert set(cell_tag.values) == set_values


@pytest.mark.parametrize(
    "entities_list, set_values",
    [
        ([(1, lambda x: x[0] <= 1.0, False)], {1}),
        ([(2, lambda x: x[0] <= 1.0, False)], {2}),
        (
            [
                (1, lambda x: x[0] <= 1.0, False),
                (2, lambda x: x[0] >= 0.0, False),
            ],
            {2},
        ),
    ],
)
def test_create_facet_tags_all_on_boundary_False(entities_list, set_values):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    facet_tag = scifem.create_entity_markers(
        mesh, mesh.topology.dim - 1, entities_list=entities_list
    )
    im = mesh.topology.index_map(mesh.topology.dim - 1)

    assert facet_tag.dim == mesh.topology.dim - 1
    assert facet_tag.indices.shape[0] == im.size_local + im.num_ghosts
    assert facet_tag.values.shape[0] == im.size_local + im.num_ghosts
    assert facet_tag.values.dtype == np.int32
    assert set(facet_tag.values) == set_values


@pytest.mark.parametrize(
    "entities_list, set_values",
    [
        ([(1, lambda x: x[0] <= 1.0, True)], {1}),
        ([(2, lambda x: x[0] <= 1.0, True)], {2}),
        (
            [
                (1, lambda x: x[0] <= 1.0, True),
                (2, lambda x: x[0] >= 0.0, True),
            ],
            {2},
        ),
    ],
)
def test_create_facet_tags_all_on_boundary_True(entities_list, set_values):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    facet_tag = scifem.create_entity_markers(
        mesh, mesh.topology.dim - 1, entities_list=entities_list
    )

    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facet_indices = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    assert facet_tag.dim == mesh.topology.dim - 1
    assert facet_tag.indices.shape[0] == len(facet_indices)
    assert facet_tag.values.shape[0] == len(facet_indices)
    assert facet_tag.values.dtype == np.int32
    assert set(facet_tag.values) == set_values


@pytest.mark.parametrize(
    "entities_list",
    [
        ([(1, lambda x: x[0] < 0.0)]),
        ([(1, lambda x: x[0] < 0.0)]),
        ([]),
    ],
)
def test_create_celltags_empty(entities_list):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    cell_tag = scifem.create_entity_markers(mesh, mesh.topology.dim, entities_list=entities_list)

    assert cell_tag.dim == mesh.topology.dim
    assert cell_tag.indices.shape[0] == 0
    assert cell_tag.values.shape[0] == 0
    assert cell_tag.values.dtype == np.int32
    assert set(cell_tag.values) == set()


import pytest


@pytest.mark.parametrize("edim", [0, 1, 2, 3])
def test_submesh_meshtags(edim):
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        3,
        4,
        7,
        cell_type=dolfinx.cpp.mesh.CellType.tetrahedron,
        ghost_mode=dolfinx.cpp.mesh.GhostMode.shared_facet,
    )

    mesh.topology.create_entities(edim)
    emap = mesh.topology.index_map(edim)

    # Put every second owned cell in submesh
    num_cells_local = emap.size_local + emap.num_ghosts
    subset_cells_local = np.arange(num_cells_local, dtype=np.int32)[::2]
    cell_communicator = dolfinx.la.vector(emap, 1)
    cell_communicator.array[subset_cells_local] = 1
    cell_communicator.scatter_reverse(dolfinx.la.InsertMode.add)
    cell_communicator.scatter_forward()
    subset_cells = np.flatnonzero(cell_communicator.array).astype(np.int32)
    submesh, entity_to_parent, vertex_to_parent, _ = dolfinx.mesh.create_submesh(
        mesh, edim, subset_cells
    )

    # Create meshtags on the parent mesh
    for i in range(edim + 1):
        mesh.topology.create_entities(i)
        parent_e_map = mesh.topology.index_map(i)
        num_parent_entities = parent_e_map.size_local + parent_e_map.num_ghosts
        values = parent_e_map.local_range[0] + np.arange(num_parent_entities, dtype=np.int32)
        entity_communicator = dolfinx.la.vector(parent_e_map, 1)
        entity_communicator.array[:] = values
        entity_communicator.scatter_forward()
        parent_tag = dolfinx.mesh.meshtags(
            mesh,
            i,
            np.arange(num_parent_entities, dtype=np.int32),
            entity_communicator.array.astype(np.int32),
        )

        sub_tag, sub_entity_to_parent = scifem.mesh.transfer_meshtags_to_submesh(
            parent_tag, submesh, vertex_to_parent, entity_to_parent
        )
        submesh.topology.create_connectivity(i, edim)

        midpoints = dolfinx.mesh.compute_midpoints(submesh, i, sub_tag.indices)
        mesh.topology.create_connectivity(i, mesh.topology.dim)
        parent_midpoints = dolfinx.mesh.compute_midpoints(mesh, i, parent_tag.indices)

        np.testing.assert_allclose(midpoints, parent_midpoints[sub_entity_to_parent])
