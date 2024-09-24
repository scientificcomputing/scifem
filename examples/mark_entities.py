# # Entity markers
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# DOLFINx gives you full control for marking entities.
# However, sometimes this can feel a bit repetative.
# In this example we will show how to use {py:func}`scifem.create_entity_markers`.

from mpi4py import MPI
import dolfinx
import numpy as np

# We start by creating a simple unit square

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 60, 60)

# Next, we want to mark some of the cells in our domain.
# We create three marker functions below.
# Each of them takes in a ``(3, num_points)`` array, and returns a boolean array of size ``num_points``.


def left(x):
    return x[0] < 0.2


def right(x):
    return x[0] > 0.9


def inner(x):
    # We use numpy bit operator `&` for "and"
    return (x[0] > 0.3) & (x[0] < 0.7)


# We want to mark these entities with  with unique integers 1, 3 and 7.

from scifem import create_entity_markers

cell_tag = create_entity_markers(mesh, mesh.topology.dim, [(1, left), (3, right), (7, inner)])

# Next we can plot these marked entities

import pyvista

pyvista.start_xvfb()
vtk_grid = dolfinx.plot.vtk_mesh(mesh, cell_tag.dim, cell_tag.indices)
grid = pyvista.UnstructuredGrid(*vtk_grid)
grid.cell_data["Marker"] = cell_tag.values

# Create plotter

plotter = pyvista.Plotter()
plotter.add_mesh(grid)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()


# We can also mark lower order entities, such as facets


def circle(x):
    return x[0] ** 2 + x[1] ** 2 <= 0.16**2


def top(x):
    return x[1] > 0.9


facet_tags = create_entity_markers(mesh, mesh.topology.dim - 1, [(2, top), (7, circle)])


facet_grid = dolfinx.plot.vtk_mesh(mesh, facet_tags.dim, facet_tags.indices)

fgrid = pyvista.UnstructuredGrid(*facet_grid)
fgrid.cell_data["Marker"] = facet_tags.values

fplotter = pyvista.Plotter()
fplotter.add_mesh(fgrid)
fplotter.view_xy()
if not pyvista.OFF_SCREEN:
    fplotter.show()

# We can also exclude interior facets by adding `on_boundary: True` (by default this is set to False).

boundary_facet_tags = create_entity_markers(
    mesh, mesh.topology.dim - 1, [(2, top, True), (7, circle, False)]
)


boundary_grid = dolfinx.plot.vtk_mesh(mesh, boundary_facet_tags.dim, boundary_facet_tags.indices)

bfgrid = pyvista.UnstructuredGrid(*boundary_grid)
bfgrid.cell_data["Marker"] = boundary_facet_tags.values

bfplotter = pyvista.Plotter()
bfplotter.add_mesh(bfgrid)
bfplotter.view_xy()
if not pyvista.OFF_SCREEN:
    bfplotter.show()
