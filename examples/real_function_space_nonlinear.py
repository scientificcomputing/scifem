# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb,md:myst
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: scifem-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Real function spaces (non linear)
#
# Author: Remi Delaporte-Mathurin, Jørgen S. Dokken
#
# License: MIT
#
# In this example we will show how to use the "real" function space to solve
# a non linear problem.
#
# A circular gas cavity inside a solid domain has an initial partial pressure $P_b$. The concentration $u$ on the cavity surface follows Henry's solubility law: the concentration is proportional to the partial pressure $P_b$.
#
# As particles leave the cavity by solution/diffusion, the partial pressure starts to decrease - affecting in turn the concentration on the cavity surface.
#
#
# ## Mathematical formulation
#  The problem at hand is:
# Find $u \in H^1(\Omega)$ such that
#
# $$
# \begin{align}
# \Delta u &= 0 \quad \text{in } \Omega, \\
# u &= 0 \quad \text{on } \Gamma_1, \\
# u &= K P_b \quad \text{on } \Gamma_b, \\
# \end{align}
# $$
#
# where $P_b$ is the partial pressure inside the cavity region. The temporal evolution of $P_b$ is governed by:
#
# $$
# \frac{dP_b}{dt} = \frac{e}{V} \int_{\Gamma_b} -D \nabla c \cdot \mathbf{n} dS
# $$
#
#
# We write the weak formulation with $v$ and $w$ suitable test functions:
#
# $$
# \begin{align}
# \int_\Omega  \frac{u - u_n}{\Delta t} \cdot v~\mathrm{d}x + \int_\Omega \nabla u \cdot \nabla v~\mathrm{d}x &= 0\\
# \int_\Omega  \frac{P_b - P_{b_n}}{\Delta t} \cdot w~\mathrm{d}x &= \frac{e}{V} \int_{\Gamma_b} -D \nabla c \cdot \mathbf{n} ~dS ~w.
# \end{align}
# $$
#

# %% [markdown]
# ## Implementation
# We start by import the necessary modules
# ```{admonition} Clickable functions/classes
# Note that for the modules imported in this example, you can click on the function/class
# name to be redirected to the corresponding documentation page.
# ```

# %%
from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import gmsh as gmshio
import numpy as np
from scifem import create_real_functionspace, assemble_scalar
import ufl

import gmsh

# %% [markdown]
# We generate the mesh using GMSH:

# %%
gmsh.initialize()

L = 2
H = L
c_x = c_y = L/2
r = 0.05
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

solid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert len(volumes) == 1
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], solid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], solid_marker, "Solid")

wall_marker, obstacle_marker = 2, 3
walls, obstacle = [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            walls.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            walls.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(
            center_of_mass, [L / 2, 0, 0]
        ):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")


res_min = r / 3
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)


if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
ft = mesh_data.facet_tags
ft.name = "Facet markers"

# %%
from dolfinx import plot
import pyvista

pyvista.set_jupyter_backend("html")

tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(mesh, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("mesh.png")

fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(mesh, fdim, ft.indices)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = ft.values
grid.set_active_scalars("Facet Marker")
p.view_xy()
p.add_mesh(grid, show_edges=True)

if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_markers.png")

# %% [markdown]
# We create two functionspaces `V` and `R` for $u$ and $P_b$, respectively, as well as a `ufl.MixedFunctionSpace` for test functions.

# %%
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
R = create_real_functionspace(mesh)

W = ufl.MixedFunctionSpace(V, R)

# %% [markdown]
# We then create appropriate functions and test functions:

# %%
u = dolfinx.fem.Function(V)
u_n = dolfinx.fem.Function(V)
pressure = dolfinx.fem.Function(R)
pressure_n = dolfinx.fem.Function(R)

v, pressure_v = ufl.TestFunctions(W)


# %% [markdown]
# Let's define the problems constants:

# %%
dt = dolfinx.fem.Constant(mesh, 0.1)
K_S = dolfinx.fem.Constant(mesh, 2.0)
e = dolfinx.fem.Constant(mesh, 2.0)
volume = dolfinx.fem.Constant(mesh, 40.0)
P_b_initial = dolfinx.fem.Constant(mesh, 3.5)
u_out = dolfinx.fem.Constant(mesh, 0.0)

# %% [markdown]
# ```{note}
# If `dt` is too large, the problem becomes unstable
# ```
#
#
# We can now define the initial condition and boundary conditions:

# %%
pressure_ini_expr = dolfinx.fem.Expression(
    P_b_initial, R.element.interpolation_points
)
pressure_n.interpolate(pressure_ini_expr)
pressure.x.array[:] = pressure_n.x.array[:]

bc_bubble_expr = dolfinx.fem.Expression(K_S * pressure, V.element.interpolation_points)
u_bc_bubble = dolfinx.fem.Function(V)
u_bc_bubble.interpolate(bc_bubble_expr)

dofs_boundary = dolfinx.fem.locate_dofs_topological(V, fdim, ft.indices[ft.values == wall_marker])
dofs_bubble = dolfinx.fem.locate_dofs_topological(V, fdim, ft.indices[ft.values == obstacle_marker]
)

bc_bubble = dolfinx.fem.dirichletbc(u_bc_bubble, dofs_bubble)
bc_boundary = dolfinx.fem.dirichletbc(u_out, dofs_boundary, V)

# %% [markdown]
# Next we define the variational formulation and call `ufl.extract_blocks` to form the blocked formulations:

# %%
n = ufl.FacetNormal(mesh)
flux = ufl.inner(ufl.grad(u), n)

ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

F = (u - u_n) / dt * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
F += (pressure - pressure_n) / dt * pressure_v * ufl.dx + e / volume * flux * pressure_v * ds(3)

forms = ufl.extract_blocks(F)

# %% [markdown]
# We can now create a nonlinear solver with the blocked formulations and the functions `u` and `pressure` as a list:

# %%
solver = NonlinearProblem(
    forms,
    [u, pressure],
    bcs=[bc_boundary, bc_bubble],
    petsc_options_prefix="bubble",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_monitor": None,
    },
)

# %% [markdown]
# Set up transient pyvista visualisation:

# %%
import matplotlib as mpl
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif", fps=10)

grid.point_data["u"] = u.x.array
warped = grid.warp_by_scalar("u", factor=0.1)

viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1,
)

renderer = plotter.add_mesh(
    warped,
    show_edges=True,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
    clim=[0, max(u_bc_bubble.x.array)],
)

# %% [markdown]
# Time stepping loop:

# %%
t = 0
t_final = 15

times = []
all_pressures = []
outgassing_fluxes = []


while t < t_final:
    t += dt.value
    times.append(t)
    print(f"Solving at time {t:.2f}")

    # Solve the problem
    (u, pressure) = solver.solve()

    # Update previous solution
    u_n.x.array[:] = u.x.array[:]
    pressure_n.x.array[:] = pressure.x.array[:]

    # Update bubble BC
    u_bc_bubble.interpolate(bc_bubble_expr)
    all_pressures.append(pressure.x.array[0].copy())

    # Update plot
    new_warped = grid.warp_by_scalar("u", factor=0.1)
    warped.points[:, :] = new_warped.points
    warped.point_data["u"][:] = u.x.array
    plotter.write_frame()

    # compute outgassing flux
    outgassing_flux = -assemble_scalar(flux * ds(2))
    outgassing_fluxes.append(outgassing_flux)
plotter.close()

# %% [markdown]
# ![title](u_time.gif)

# %% [markdown]
# We see that, as expected, the partial pressure $P_b$ decreases with time.

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)


axs[0].scatter([0], [P_b_initial.value], color="red", label="Initial Pressure")
axs[0].plot(times, all_pressures)
axs[0].set_ylabel("Pressure")
axs[0].set_ylim(bottom=0)

axs[1].plot(times, outgassing_fluxes)
axs[1].set_ylabel("Outgassing Flux")
axs[1].set_ylim(bottom=0)

plt.xlabel("Time")
plt.show()
