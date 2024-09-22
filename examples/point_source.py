# # Point sources in DOLFINx
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT

# In this example, we will illustrate how to apply a point source to a Poisson problem.
#
# $$
# \begin{align}
# -\nabla^2 u &= f \quad \text{in } \Omega, \\
# f &= \gamma \delta(x - p) \quad \text{in } \Omega,
# \end{align}
# $$
#
# with homogeneous Dirichlet boundary conditions.
#
# Using integration by parts, we obtain the variational problem:
#
# Find $u_h\in V_0$ such that
#
# $$
# \begin{align}
# \int_\Omega \nabla u_h \cdot \nabla v \mathrm{d}x &= \int_\Omega \gamma\delta(x-p) v \mathrm{d}x
# = \gamma v(p) \quad \forall v \in V_0,
# \end{align}
# $$

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl
import numpy as np
import scifem

# We start by creating the mesh and function space.
# To illustrate the usage the point source function, we will
# consider a vector space with to components, and we will
# only apply the point source to the second component of our problem.

N = 40
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
domain.name = "mesh"
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim-1, tdim)

# Create the two-component vector space.
# The solution in the first component will be 0.

value_shape = (2,)
V = dolfinx.fem.functionspace(domain, ("Lagrange", 2, value_shape))

# Create standard variational form

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

# Create Dirichlet boundary conditions

boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
dofs = dolfinx.fem.locate_dofs_topological(V, 1, boundary_facets)
u_bc = dolfinx.fem.Function(V)
u_bc.x.array[:] = 0
bc = dolfinx.fem.dirichletbc(u_bc, dofs)


# We are now ready to apply the point source.
# First, we create the right hand side vector for the problem, and initialize it as zero.

b = dolfinx.fem.Function(V)
b.x.array[:] = 0

# Secondly we define the point sources we want to apply.
#
# ```{warning}
# Note that if running in parallel, we only supply the points on a single process.
# The other process gets an empty array.
# ```

geom_dtype = domain.geometry.x.dtype
if domain.comm.rank == 0:
    points = np.array([[0.68, 0.362, 0],
                       [0.14, 0.213, 0]], dtype=geom_dtype)
else:
    points = np.zeros((0, 3), dtype=geom_dtype)

# Next, we create the point source object and apply it to the right hand side vector.

gamma = 1.
point_source = scifem.PointSource(V.sub(1), points, magnitude=gamma)
point_source.apply_to_vector(b)

# We can continue to solve our variational problem as usual

a_compiled = dolfinx.fem.form(a)
dolfinx.fem.petsc.apply_lifting(b.vector, [a_compiled], [[bc]])
b.x.scatter_reverse(dolfinx.la.InsertMode.add)
dolfinx.fem.petsc.set_bc(b.vector, [bc])
b.x.scatter_forward()

A = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=[bc])
A.assemble()

# Set up a direct solver an solve the linear system

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.getPC().setFactorSolverType("mumps")


uh = dolfinx.fem.Function(V)
ksp.solve(b.vector, uh.vector)
uh.x.scatter_forward()

# We can now plot the solution

import pyvista

pyvista.start_xvfb()
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(uh)] = uh.x.array.real.reshape((geometry.shape[0], len(uh)))

# Create a point cloud of glyphs

function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.2)

# Create a pyvista-grid for the mesh

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(domain, domain.topology.dim))

# Create plotter

plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
