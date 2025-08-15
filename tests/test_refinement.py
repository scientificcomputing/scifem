from mpi4py import MPI
import numpy as np
import dolfinx
import scifem

import pytest


def test_fix_overconstrained_cells_boundary():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    tdim = mesh.topology.dim 

    # Cell meshtag to be transferred
    # FIXME : Needs to be non-zero to check the transfer
    num_cells = mesh.topology.index_map(tdim).size_local
    cell_meshtag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, np.arange(0, num_cells), np.zeros(num_cells, dtype=np.int32))
    
    # Facet meshtag to be transferred
    # FIXME : Needs to be non-zero to check the transfer
    mesh.topology.create_connectivity(tdim, tdim - 1)
    num_facets = mesh.topology.index_map(tdim - 1).size_local
    facet_meshtag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, np.arange(0, num_facets), np.zeros(num_facets, dtype=np.int32))

    # Apply the fix_overconstrained_cells function
    fixed_mesh, cell_meshtag, facet_meshtag, cell_tags = scifem.fix_overconstrained_cells(mesh, cell_meshtag, facet_meshtag)
    
    ## TODO : Check splitted facets
    ## TODO : Check transfer of cell_meshtag and facet_meshtag
    assert True
    
def test_fix_overconstrained_cells_interface():
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    tdim = mesh.topology.dim

    # Define a meshtags to mark the inner cube
    inner_cube_entities = dolfinx.mesh.locate_entities(
        mesh, tdim,
        lambda x: (
            np.greater_equal(x[0], 0.2) & np.less_equal(x[0], 0.8) &
            np.greater_equal(x[1], 0.2) & np.less_equal(x[1], 0.8) &
            np.greater_equal(x[2], 0.2) & np.less_equal(x[2], 0.8)        )
    )

    # Marking inner cube entities with inner_tag, indicating we need to repair overconstrained cells from inner_cube
    inner_tag = 10
    num_cells = mesh.topology.index_map(tdim).size_local
    inner_cube_values = np.zeros(num_cells, dtype=np.int32)
    inner_cube_values[inner_cube_entities] = inner_tag  # Mark the inner cube cells
    inner_meshtag = dolfinx.mesh.meshtags(mesh, tdim, np.arange(num_cells, dtype=np.int32), inner_cube_values)

    # Cell meshtag to be transferred
    num_cells = mesh.topology.index_map(tdim).size_local
    cell_meshtag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, np.arange(0, num_cells), np.zeros(num_cells, dtype=np.int32))
    
    # Facet meshtag to be transferred
    mesh.topology.create_connectivity(tdim, tdim - 1)
    num_facets = mesh.topology.index_map(tdim - 1).size_local
    facet_meshtag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, np.arange(0, num_facets), np.zeros(num_facets, dtype=np.int32))

    # Apply the fix_overconstrained_cells function
    fixed_mesh, inner_cube_meshtag, facet_meshtag, cell_tags = scifem.fix_overconstrained_cells(mesh, cell_meshtag, facet_meshtag, cell_tag=inner_meshtag)

    ## TODO : Check splitted facets
    ## TODO : Check transfer of 
    assert True
