# # Nonlinear elasticity with blocked Newton solver
#
# Author: Henrik N. T. Finsberg
#
# SPDX-License-Identifier: MIT

# In this example we will solve a nonlinear elasticity problem using a blocked Newton solver.
# We consider a unit cube domain $\Omega = [0, 1]^3$ with Dirichlet boundary conditions on the left face and traction force on the right face, and we seek a displacement field $\mathbf{u}: \Omega \to \mathbb{R}^3$ that solves the momentum balance equation
#
# $$
# \begin{align}
# \nabla \cdot \mathbf{P} = 0, \quad \mathbf{X} \in \Omega, \\
# \end{align}
# $$
#
# where $\mathbf{P}$ is the first Piola-Kirchhoff stress tensor, and $\mathbf{X}$ is the reference configuration. We consider a Neo-Hookean material model, where the strain energy density is given by
#
# $$
# \begin{align}
# \psi = \frac{\mu}{2}(\text{tr}(\mathbf{C}) - 3),
# \end{align}
# $$
#
# and the first Piola-Kirchhoff stress tensor is given by
#
# $$
# \begin{align}
# \mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}
# \end{align}
# $$
#
# where $\mathbf{F} = \nabla \mathbf{u} + \mathbf{I}$ is the deformation gradient, $\mathbf{C} = \mathbf{F}^T \mathbf{F}$ is the right Cauchy-Green tensor, $\mu$ is the shear modulus, and $p$ is the pressure.
# We also enforce the incompressibility constraint
#
# $$
# \begin{align}
# J = \det(\mathbf{F}) = 1,
# \end{align}
# $$
#
# so that the total Lagrangian is given by
#
# $$
# \begin{align}
# \mathcal{L}(\mathbf{u}, p) = \int_{\Omega} \psi \, dx - \int_{\partial \Omega} t \cdot \mathbf{u} \, ds + \int_{\Omega} p(J - 1) \, dx.
# \end{align}
# $$
#
# Here $t$ is the traction force which is set to $10$ on the right face of the cube and $0$ elsewhere.
# The Euler-Lagrange equations for this problem are given by: Find $\mathbf{u} \in V$ and $p \in Q$ such that
#
# $$
# \begin{align}
# D_{\delta \mathbf{u} } \mathcal{L}(\mathbf{u}, p) = 0, \quad \forall \delta \mathbf{u} \in V, \\
# D_{\delta p} \mathcal{L}(\mathbf{u}, p) = 0, \quad \forall \delta p \in Q,
# \end{align}
# $$
#
# where $V$ is the displacement space and $Q$ is the pressure space. For this we select $Q_2/P_1$ elements i.e second order Lagrange elements for $\mathbf{u}$ and [first order discontinuous polynomial cubical elements](https://defelement.com/elements/examples/quadrilateral-dpc-1.html) for $p$, which is a stable element for incompressible elasticity {cite}`auricchio2013approximation`.
# Note also that the Euler-Lagrange equations can be derived automatically using `ufl`.
#
#

import logging
from mpi4py import MPI
import numpy as np
import ufl
import dolfinx
import scifem

# Initialize logging and set log level to info

logging.basicConfig(level=logging.INFO)

# We create the mesh and the function spaces

mesh = dolfinx.mesh.create_unit_cube(
    MPI.COMM_WORLD, 3, 3, 3, dolfinx.mesh.CellType.hexahedron, dtype=np.float64
)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
Q = dolfinx.fem.functionspace(mesh, ("DPC", 1,))

# And the test and trial functions
#

v = ufl.TestFunction(V)
q = ufl.TestFunction(Q)
du = ufl.TrialFunction(V)
dp = ufl.TrialFunction(Q)
u = dolfinx.fem.Function(V)
p = dolfinx.fem.Function(Q)


# Next we create the facet tags for the left and right faces

def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], 1)


facet_tags = scifem.create_entity_markers(
    mesh, mesh.topology.dim - 1, [(1, left, True), (2, right, True)]
)

# We create the Dirichlet boundary conditions on the left face

facets_left = facet_tags.find(1)
dofs_left = dolfinx.fem.locate_dofs_topological(V, 2, facets_left)
u_bc_left = dolfinx.fem.Function(V)
u_bc_left.x.array[:] = 0
bc = dolfinx.fem.dirichletbc(u_bc_left, dofs_left)

# Define the deformation gradient, right Cauchy-Green tensor, and invariants of the deformation tensors

d = len(u)
I = ufl.Identity(d)             # Identity tensor
F = I + ufl.grad(u)             # Deformation gradient
C = F.T*F                       # Right Cauchy-Green tensor
I1 = ufl.tr(C)                  # First invariant of C
J  = ufl.det(F)                 # Jacobian of F

# Traction for to be applied on the right face

t = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(10.0))

N = ufl.FacetNormal(mesh)

# Material parameters and strain energy density
mu = dolfinx.fem.Constant(mesh, 10.0)
psi = (mu / 2)*(I1 - 3)

# We for the total Lagrangian

L = psi*ufl.dx - ufl.inner(t * N, u)*ufl.ds(subdomain_data=facet_tags, subdomain_id=2)  + p * (J - 1) * ufl.dx

# and take the first variation of the total Lagrangian to obtain the residual

r_u = ufl.derivative(L, u, v)
r_p = ufl.derivative(L, p, q)
R = [r_u, r_p]

# We do the same for the second variation to obtain the Jacobian

K = [
    [ufl.derivative(r_u, u, du), ufl.derivative(r_u, p, dp)],
    [ufl.derivative(r_p, u, du), ufl.derivative(r_p, p, dp)],
]


# Now we can create the Newton solver and solve the problem

petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
solver = scifem.NewtonSolver(R, K, [u, p], max_iterations=25, bcs=[bc], petsc_options=petsc_options)

# We can also set a callback function that is called before and after the solve, which takes the solver object as input


def pre_solve(solver: scifem.NewtonSolver):
    print(f"Starting solve with {solver.max_iterations} iterations")


def post_solve(solver: scifem.NewtonSolver):
    print(f"Solve completed in with correction norm {solver.dx.norm(0)}")


solver.set_pre_solve_callback(pre_solve)
solver.set_post_solve_callback(post_solve)

solver.solve()

# Finally, we can visualize the solution using `pyvista`

import pyvista
pyvista.start_xvfb()
p = pyvista.Plotter()
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
linear_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
p.add_mesh(linear_grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
p.add_mesh(warped, show_edges=False)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("displacement.png")


# # References
# ```{bibliography}
# ```
#
