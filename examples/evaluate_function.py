# # Evaluating a function at a point
#
# Author: Henrik N.T. Finsberg
#
# SPDX-License-Identifier: MIT
#
# In this example we will show a how to to scifem for evaluating a function at a point.
# Note that the implementation is based on the approach outlined [here](https://jsdokken.com/FEniCS-workshop/src/deep_dive/expressions.html#evalation-at-a-point),
# and users are encouraged to read this for more details.

# Let us start by creating a rectangle mesh and a function space.

from mpi4py import MPI
import numpy as np
import dolfinx

from scifem import evaluate_function

comm = MPI.COMM_WORLD
Lx = Ly = 2.0
nx = ny = 10

mesh = dolfinx.mesh.create_rectangle(
    comm=comm,
    points=[np.array([0.0, 0.0]), np.array([Lx, Ly])],
    n=[nx, ny],
    cell_type=dolfinx.mesh.CellType.triangle
)

V = dolfinx.fem.functionspace(mesh, ("P", 1))
u = dolfinx.fem.Function(V)

# Now let us interpolate a function $f(x, y) = x + 2y$ into the function space.
# and use this as an example for evaluating the function at a set of points.

f = lambda x: x[0] + 2 * x[1]
u.interpolate(f)

# Let us pick a few points to evaluate the function at.

points = np.array([[0.0, 0.0], [0.2, 0.2], [0.5, 0.5], [0.7, 0.2]])

# The expected values of the function at the points are

exact = np.array(f(points.T)).T
print(exact)

# We can now evaluate the function at the points using the `evaluate_function` function.

u_values = evaluate_function(u, points)
print(u_values)

# Finally, we plot the function, and label each point with the value we evaluated there

import pyvista

grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
grid.point_data["u"] = u.x.array.real
points_3d = np.zeros((points.shape[0], 3))
points_3d[:, :2] = points

plotter = pyvista.Plotter()
plotter.add_mesh(grid, scalars="u", show_edges=True, cmap="viridis")
plotter.add_point_labels(
    points_3d,
    [f"{value:.2f}" for value in np.asarray(u_values).real.flatten()],
    point_size=14,
    point_color="red",
    render_points_as_spheres=True,
    font_size=16,
    always_visible=True,
)
plotter.view_xy()
plotter.export_html("evaluate_function.html")
plotter.show(screenshot="evaluate_function.png")
