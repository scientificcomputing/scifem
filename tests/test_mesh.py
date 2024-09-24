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
