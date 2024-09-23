from mpi4py import MPI
import numpy as np
import dolfinx
import scifem

import pytest


@pytest.mark.parametrize(
    "entities_list, set_values",
    [
        ([{"tag": 1, "locator": lambda x: x[0] <= 1.0}], {1}),
        ([{"tag": 2, "locator": lambda x: x[0] <= 1.0}], {2}),
        (
            [
                {"tag": 1, "locator": lambda x: x[0] <= 1.0},
                {"tag": 2, "locator": lambda x: x[0] >= 0.0},
            ],
            {2},
        ),
    ],
)
def test_create_meshtags_celltags_all(entities_list, set_values):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    cell_tag = scifem.create_meshtags(mesh, mesh.topology.dim, entities_list=entities_list)

    im = mesh.topology.index_map(mesh.topology.dim)
    assert cell_tag.dim == mesh.topology.dim
    assert cell_tag.indices.shape[0] == im.size_local + im.num_ghosts
    assert cell_tag.values.shape[0] == im.size_local + im.num_ghosts
    assert cell_tag.values.dtype == np.int32
    assert set(cell_tag.values) == set_values


@pytest.mark.parametrize(
    "entities_list",
    [
        ([{"tag": 1, "locator": lambda x: x[0] < 0.0}]),
        ([]),
    ],
)
def test_create_meshtags_celltags_empty(entities_list):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    cell_tag = scifem.create_meshtags(mesh, mesh.topology.dim, entities_list=entities_list)

    assert cell_tag.dim == mesh.topology.dim
    assert cell_tag.indices.shape[0] == 0
    assert cell_tag.values.shape[0] == 0
    assert cell_tag.values.dtype == np.int32
    assert set(cell_tag.values) == set()
