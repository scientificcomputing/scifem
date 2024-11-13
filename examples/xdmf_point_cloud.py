# # Visualizing quadrature functions as point clouds
#
# Author: Henrik N.T. Finsberg
#
# SPDX-License-Identifier: MIT
#
# Quadrature functions are not possible to visualize directly in ParaView, as they are not defined on a mesh.
# However, we can visualize them as point clouds.
# In this example we will show how you can use `scifem` to save your quadrature fuctions as XDMF files,
# which can be loaded into ParaView for visualization.
#
# ```{note}
# This demo requires a backend for writing HDF5 files. Please install `scifem` with either `scifem[adios2]` or `scifem[h5py]`.
# ```

# First, we import the necessary modules.

import logging
from mpi4py import MPI
import numpy as np
import basix
import dolfinx
import scifem

# First initialize logging
logging.basicConfig(level=logging.INFO)

# Now let's create a quadrature function on a unit square mesh. We will use the `basix.ufl` module to create the quadrature element.

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, dolfinx.mesh.CellType.triangle, dtype=np.float64)
el = basix.ufl.quadrature_element(
    scheme="default", degree=3, cell=mesh.ufl_cell().cellname(), value_shape=()
)
V = dolfinx.fem.functionspace(mesh, el)
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))


# One option that already exists with the current DOLFINx/Pyvista API is to plot them as points


import pyvista

pyvista.start_xvfb()
plotter = pyvista.Plotter()
plotter.add_points(
    V.tabulate_dof_coordinates(),
    scalars=u.x.array,
    render_points_as_spheres=True,
    point_size=20,
    show_scalar_bar=False,
)
if not pyvista.OFF_SCREEN:
    plotter.show()


# Using `scifem`, we can write the point cloud data to an XDMFFile that can be opened with Paraview.

with scifem.xdmf.XDMFFile("point_cloud.xdmf", [u]) as xdmf:
    xdmf.write(0.0)

# The point cloud can now be loaded into ParaView for visualization, by selecting "Point Gaussian" as the representation.
# ![Point cloud in ParaView](../docs/_static/point_cloud.png)

# We can write any `dolfinx.fem.FunctionSpace` that has support for `tabulate_dof_coordinates` to a point cloud.
# For example, we can create a higher order Lagrange space, and write two functions to file.

Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 3, (2, )))
q_1 = dolfinx.fem.Function(Q, name="q_sin")
q_1.interpolate(lambda x: (np.sin(np.pi * x[0]),  np.sin(np.pi * x[1])))
q_2 = dolfinx.fem.Function(Q, name="q_cos")
q_2.interpolate(lambda x: (np.cos(np.pi * x[0]),  np.cos(np.pi * x[1])))

# We write these two functions to file as illustrated above

with scifem.xdmf.XDMFFile("point_cloud_lagrange.xdmf", [q_1, q_2]) as xdmf:
    xdmf.write(0.0)

# yielding the following point clouds in ParaView after applying glyphs.
# ![Point cloud in ParaView](../docs/_static/cos_sin_pointcloud.png)

# If you have time dependent data, you can write multiple time steps to the same file using e.g

with scifem.xdmf.XDMFFile("point_cloud_lagrange.xdmf", [q_1, q_2]) as xdmf:
    xdmf.write(0.0)
    q_1.interpolate(lambda x: (-x[0], x[1]))
    q_2.interpolate(lambda x: (x[0], -x[1]))
    xdmf.write(0.3)

# If you need to keep the file open for longer, you can use the following syntax

xdmf = scifem.xdmf.XDMFFile("point_cloud_lagrange.xdmf", [q_1, q_2])
xdmf.write(0.0)
q_1.interpolate(lambda x: (-x[0], x[1]))
q_2.interpolate(lambda x: (x[0], -x[1]))
xdmf.write(0.3)
xdmf.close()
