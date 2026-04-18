# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,md:myst,py:percent
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
# In this example we will show:
# - How to define a submesh on a subset of facets
# - How to use the "real" function space on this submesh
# - How to solve a problem coupling this real space with the bulk using
#   a nonlinear solver.
#
# A circular gas cavity inside a solid domain has an initial partial
# pressure $P_b$.
# The concentration $u$ on the cavity surface follows Henry's solubility law:
# the concentration is proportional to the partial pressure $P_b$.
#
# As particles leave the cavity by solution/diffusion, the partial pressure
# starts to decrease - affecting in turn the concentration on the
# cavity surface.
#
#
# ## Mathematical formulation
#  The problem at hand is:
# Find $u \in H^1(\Omega)$ such that
#
# $$
# \begin{align}
# \frac{du}{dt}&= \Delta u + f\quad \text{in } \Omega, \\
# u &= 0 \quad \text{on } \Gamma_1, \\
# u &= K P_b \quad \text{on } \Gamma_b,
# \end{align}
# $$
#
# where $P_b\in \mathbb{R}$ is the partial pressure inside the cavity region.
# The temporal evolution of $P_b$ is governed by:
#
# $$
# \frac{dP_b}{dt} = A \int_{\Gamma_b} -\nabla u \cdot \mathbf{n} dS
# $$
#
#
# We write the weak formulation with $v$ and $w$ suitable test functions:
#
# $$
# \begin{align}
# \int_\Omega  \frac{u - u_n}{\Delta t} \cdot v~\mathrm{d}x
# + \int_\Omega \nabla u \cdot \nabla v~\mathrm{d}x &= 0\\
# \int_\Omega  \frac{P_b - P_{b_n}}{\Delta t} \cdot w~\mathrm{d}x
# &= A \int_{\Gamma_b} -\nabla u \cdot \mathbf{n} \cdot ~w ~\mathrm{d}S.
# \end{align}
# $$
#
# ## Implementation
# We start by import the necessary modules
# ```{admonition} Clickable functions/classes
# Note that for the modules imported in this example, you can click on the function/class
# name to be redirected to the corresponding documentation page.
# ```

# %%
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import gmsh as gmshio
from dolfinx.plot import vtk_mesh
import numpy as np
from scifem import create_real_functionspace, assemble_scalar
import ufl
import pyvista
import gmsh
import matplotlib as mpl
import matplotlib.pyplot as plt


# %% [markdown]
# ### Mesh generation
# We generate the mesh using `dolfinx.mesh.create_interval`.

# %%
import numpy as np
import scipy.optimize as opt
from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from scifem import create_real_functionspace
import ufl
import matplotlib.pyplot as plt

# Options
nx = 400
l_val = 1.5
D_val = 1.0
K_H_val = 2.0
A_form_val = 1.0  # corresponds to A * k_B * T / V in the 1D model
P_0_val = 3.5
dt_val = 0.01
t_final = 2.0
alpha_penalty = 10.0

L_const = K_H_val * A_form_val

# 1D mesh
mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, nx, [0.0, l_val])

fdim = mesh.topology.dim - 1
boundaries = [(1, lambda x: np.isclose(x[0], 0.0)),
              (2, lambda x: np.isclose(x[0], l_val))]

facet_indices, facet_markers = [], []
for (marker, locator) in boundaries:
    facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
ft = dolfinx.mesh.meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# %% [markdown]
# ### Submeshes and real function space
# We create two functionspaces `V` and `R` for $u$ and $P_b$, respectively.
# The partial pressure is a single, constant value, that should be defined
# on the interface between the cavity and the fluid domain.
# This is a "real" function space, that has only one degree of freedom,
# and is not associated with any mesh entity.
# We can create such a function space using the helper function
# {py:func}`scifem.create_real_functionspace`.
# However, first we will create a submesh on the cavity surface,
# as this is where the partial pressure is defined.

# %%
cavity_surface, cavity_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, fdim, ft.find(2))
R = create_real_functionspace(cavity_surface)

# %% [markdown]
# We will solve this as a strongly coupled problem.
# To do so, we use {py:class}`ufl.MixedFunctionSpace` test functions
# that respect the block structure of the problem.

# %%
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
W = ufl.MixedFunctionSpace(V, R)

# %% [markdown]
# ## Variational formulation and nonlinear solver
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
dt = dolfinx.fem.Constant(mesh, dt_val)
K_S = dolfinx.fem.Constant(mesh, K_H_val)
A = dolfinx.fem.Constant(mesh, A_form_val)
P_b_initial = dolfinx.fem.Constant(cavity_surface, P_0_val)

# %% [markdown]
# We can now define the initial condition and boundary conditions:

# %%
pressure_ini_expr = dolfinx.fem.Expression(
    P_b_initial, R.element.interpolation_points
)
pressure_n.interpolate(pressure_ini_expr)
pressure.x.array[:] = pressure_n.x.array[:]

# Dirichlet BC on x=0
u_bc = dolfinx.fem.Function(V)
u_bc.x.array[:] = 0.0
dofs_0 = dolfinx.fem.locate_dofs_topological(V, fdim, ft.find(1))
bc_boundary = dolfinx.fem.dirichletbc(u_bc, dofs_0)

# %% [markdown]
# Next we define the variational formulation and call
# {py:func}`ufl.extract_blocks` to form the blocked formulations:

# %%
# For the boundary condition coupling pressure and concentration,
# we use a Nitsche formulation,
# see: https://jsdokken.com/dolfinx-tutorial/chapter1/nitsche.html
# for more details about Nitsche's method.

n = ufl.FacetNormal(mesh)
flux_u = D_val * ufl.grad(u)
flux_v = D_val * ufl.grad(v)
h_val = dolfinx.fem.Constant(mesh, l_val / nx)

g = K_H_val * pressure

ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

# Fully coupled Nitsche formulation
F = ufl.inner((u - u_n) / dt, v) * ufl.dx
F += ufl.inner(flux_u, ufl.grad(v)) * ufl.dx

# Boundary ODE at x=l
F += ufl.inner((pressure - pressure_n) / dt, pressure_v) * ds(2)
F += ufl.inner(A_form_val * ufl.dot(flux_u, n), pressure_v) * ds(2) # Outflux from domain into cavity decreases P_b

# Nitsche coupling at x=l
F -= ufl.inner(ufl.dot(flux_u, n), v) * ds(2)
F -= ufl.inner(u - g, ufl.dot(flux_v, n)) * ds(2)
F += (alpha_penalty / h_val) * ufl.inner(u - g, v) * ds(2)

forms = ufl.extract_blocks(F)

# %% [markdown]
# We can now create a
# {py:class}`nonlinear solver<dolfinx.fem.petsc.NonlinearProblem>`
# with the blocked formulations and the functions `u` and `pressure` as a list:

# %%
solver = NonlinearProblem(
    forms,
    [u, pressure],
    bcs=[bc_boundary],
    petsc_options_prefix="bubble",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_error_if_not_converged": True,
        "ksp_error_if_not_converged": True,
        "snes_monitor": None,
    },
    entity_maps=[cavity_map]
)

# %% [markdown]
# Time stepping loop:

# %%
t = 0.0
times = [t]
p_fem = [P_0_val]
flux_fem = [0.0]

# Evaluate the flux vector at x=0
flux_0_form = dolfinx.fem.form(ufl.inner(flux_u, n) * ds(1))

par_print = PETSc.Sys.Print
while t < t_final:
    t += dt_val
    times.append(t)
    par_print(f"Time: {t:.2f} / {t_final:.2f}")

    # Solve the problem
    solver.solve()

    # Update previous solution
    u_n.x.array[:] = u.x.array[:]
    pressure_n.x.array[:] = pressure.x.array[:]

    # Update bubble BC
    p_fem.append(pressure.x.array[0].copy())

    # n points in -x direction at x=0, so flux in +x dir is -(flux_u * n)
    F_out = dolfinx.fem.assemble_scalar(flux_0_form)
    flux_0 = mesh.comm.allreduce(F_out, op=MPI.SUM)
    flux_fem.append(-flux_0)

# %% [markdown]
# We see that, as expected, the partial pressure $P_b$ decreases with time.

# %%
def get_exact_roots(L, l_val, num_roots):
    exact_roots = []
    def root_eqn(alpha):
        return alpha * np.tan(alpha * l_val) - L

    for n_root in range(num_roots):
        start = (n_root * np.pi) / l_val + 1e-9
        end = (n_root * np.pi + np.pi/2) / l_val - 1e-9
        r_root = opt.brentq(root_eqn, start, end)
        exact_roots.append(r_root)
    return np.array(exact_roots)

alphas = get_exact_roots(L=L_const, l_val=l_val, num_roots=400)

# Analytical Model processing
time_arr = np.array(times)
p_ana = np.zeros_like(time_arr)
flux_ana = np.zeros_like(time_arr)

for i, t_ in enumerate(time_arr):
    if t_ == 0:
        p_ana[i] = P_0_val
        flux_ana[i] = 0.0
        continue
    c_l = 0.0
    J_0 = 0.0
    for alpha in alphas:
        term = np.exp(-D_val * t_ * alpha**2) / (L_const + l_val * (L_const**2 + alpha**2))
        c_l += term
        J_0 += term * alpha / np.sin(l_val * alpha)

    p_ana[i] = 2 * L_const * P_0_val * c_l
    flux_ana[i] = 2 * L_const * P_0_val * D_val * K_H_val * J_0

# Plotting the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(time_arr, p_fem, 'o', label='FEM (scifem)', alpha=0.6)
ax1.plot(time_arr, p_ana, '-', label='Analytical (Ambrosek et al.)', linewidth=2)
ax1.set_ylabel('Boundary Pressure $(P_b)$')
ax1.legend()
ax1.grid(True)

ax2.plot(time_arr, flux_fem, 'o', label='FEM (scifem)', alpha=0.6)
ax2.plot(time_arr, flux_ana, '-', label='Analytical (Ambrosek et al.)', linewidth=2)
ax2.set_ylabel('Outgassing Flux at $x=0$')
ax2.set_xlabel('Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
