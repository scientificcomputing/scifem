# # Transfer meshtags to submeshes
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# In DOLFINx, one can create sub-meshes of entities of any co-dimension $(0,\dots,\mathrm{tdim})$.
# For complex meshes, it is not easy to locate subdomains or boundaries of interest.
# In this example, we will import a mesh from GMSH, where we have marked different parts of the domain
# with different markers, and transfer these markers to a submesh.


import os
import sys
import pyvista
from scifem import transfer_meshtags_to_submesh
from mpi4py import MPI
import gmsh
import dolfinx
import numpy as np

# We start by embedding an ellipsoid within another ellipsoid using GMSH.
# For more details about creating this mesh in GMSH,
# see: [FEniCS Workshop: External meshes](https://jsdokken.com/FEniCS-workshop/src/external_mesh.html)

gmsh.initialize()

center = (0, 0, 0)
aspect_ratio = 0.5
R_i = 0.3
R_e = 0.8

inner_disk = gmsh.model.occ.addDisk(*center, R_i, aspect_ratio * R_i)
outer_disk = gmsh.model.occ.addDisk(*center, R_e, R_e)
whole_domain, map_to_input = gmsh.model.occ.fragment(
    [(2, outer_disk)], [(2, inner_disk)])
gmsh.model.occ.synchronize()
circle_inner = [idx for (dim, idx) in map_to_input[1] if dim == 2]
circle_outer = [idx for (dim, idx) in map_to_input[0]
                if dim == 2 and idx not in circle_inner]
gmsh.model.addPhysicalGroup(2, circle_inner, tag=3)
gmsh.model.addPhysicalGroup(2, circle_outer, tag=7)
inner_boundary = gmsh.model.getBoundary(
    [(2, e) for e in circle_inner], recursive=False, oriented=False)
outer_boundary = gmsh.model.getBoundary(
    [(2, e) for e in circle_outer], recursive=False, oriented=False)
interface = [idx for (dim, idx) in inner_boundary if dim == 1]
ext_boundary = [
    idx for (dim, idx) in outer_boundary if idx not in interface and dim == 1]

gmsh.model.addPhysicalGroup(1, interface, tag=12)
gmsh.model.addPhysicalGroup(1, ext_boundary, tag=15)

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(3)
gmsh.model.mesh.optimize("Netgen")

# Next, we read this mesh and corresponding markers into DOLFINx

circular_mesh, cell_marker, facet_marker = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

# We visualize the mesh and its markers

# + tags=["hide-input"]

if sys.platform == "linux" and (os.getenv("CI") or pyvista.OFF_SCREEN):
    pyvista.start_xvfb(0.05)


def plot_mesh(mesh: dolfinx.mesh.Mesh, values=None):
    """
    Given a DOLFINx mesh, create a `pyvista.UnstructuredGrid`,
    and plot it and the mesh nodes
    """
    plotter = pyvista.Plotter()
    V_linear = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    linear_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_linear))
    if mesh.geometry.cmap.degree > 1:
        ugrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
        if values is not None:
            ugrid.cell_data["Marker"] = values
        plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        ugrid = ugrid.tessellate()
        plotter.add_mesh(ugrid, show_edges=False)
        plotter.add_mesh(linear_grid, style="wireframe", color="black")

    else:
        if values is not None:
            linear_grid.cell_data["Marker"] = values
        plotter.add_mesh(linear_grid, show_edges=True)
    plotter.show_axes()
    plotter.view_xy()
    plotter.show()


plot_mesh(circular_mesh, cell_marker.values)
# -

# Next, we create a submesh, only extracting the upper half of the mesh
tdim = circular_mesh.topology.dim
submesh, cell_map, vertex_map, node_map = dolfinx.mesh.create_submesh(
    circular_mesh, tdim, dolfinx.mesh.locate_entities(circular_mesh, tdim, lambda x: x[1] > 0))

# We transfer the cell markers to the submesh

sub_cell_marker, sub_cell_map = transfer_meshtags_to_submesh(
    cell_marker, submesh, vertex_map, cell_map)

# and visualize it

# + tags=["hide-input"]
plot_mesh(submesh, sub_cell_marker.values)
# -
