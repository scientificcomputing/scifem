# # Visualizing quadrature functions as point clouds
# Quadrature functions are not possible to visualize directly in ParaView, as they are not defined on a mesh. However, we can visualize them as point clouds.
# In this example we will show how you can use `scifem` to save your quadrature fuctions as XDMF files, which can be loaded into ParaView for visualization.

# First, we import the necessary modules.

from mpi4py import MPI
import numpy as np
import basix
import dolfinx
import scifem

# Now let's create a quadrature function on a unit square mesh. We will use the `basix` module to create the quadrature element.

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, dolfinx.mesh.CellType.triangle, dtype=np.float64)
el = basix.ufl.quadrature_element(
    scheme="default", degree=3, cell=mesh.ufl_cell().cellname(), value_shape=()
)
V = dolfinx.fem.functionspace(mesh, el)
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))


# We can now plot the solution with pyvista.


# import pyvista

# pyvista.start_xvfb()
# plotter = pyvista.Plotter()
# plotter.add_points(
#     V.tabulate_dof_coordinates(),
#     scalars=u.x.array,
#     render_points_as_spheres=True,
#     point_size=20,
#     show_scalar_bar=False,
# )
# if not pyvista.OFF_SCREEN:
#     plotter.show()


# We can also save the point cloud to a file using the `scifem` module.

scifem.xdmf.create_pointcloud("point_cloud.xdmf", [u])

# The point cloud can now be loaded into ParaView for visualization, by selecting "Point Gaussian" as the representation.
# ![Point cloud in ParaView](../docs/_static/point_cloud.png)
