# # Real function spaces
#
# Author: Jørgen S. Dokken
#
# License: MIT
#
# In this example we will show how to use the "real" function space to solve
# a singular Poisson problem.
# The problem at hand is:
# Find $u \in H^1(\Omega)$ such that
# \begin{align}
# -\Delta u &= f \quad \text{in } \Omega, \\
# \frac{\partial u}{\partial n} &= g \quad \text{on } \partial \Omega, \\
# \int_\Omega u &= h.
# \end{align}
#
# ## Lagrange multiplier
# We start by considering the equivalent optimization problem:
# Find $u \in H^1(\Omega)$ such that
# \begin{align}
# \min_{u \in H^1(\Omega)} J(u) = \min_{u \in H^1(\Omega)} \frac{1}{2}\int_\Omega \vert \nabla u \cdot \nabla u \vert \mathrm{d}x - \int_\Omega f u \mathrm{d}x - \int_{\partial \Omega} g u \mathrm{d}s,
# \end{align}
# such that
# \begin{align}
# \int_\Omega u = h.
# \end{align}
# We introduce a Lagrange multiplier $\lambda$ to enforce the constraint:
# \begin{align}
# \min_{u \in H^1(\Omega), \lambda\in \mathbb{R}} \mathcal{L}(u, \lambda) = \min_{u \in H^1(\Omega), \lambda\in \mathbb{R}} J(u) + \lambda (\int_\Omega u \mathrm{d}x-h).
# \end{align}
# We then compute the optimality conditions for the problem above
# \begin{align}
# \frac{\partial \mathcal{L}}{\partial u}[\delta u] &= \int_\Omega \nabla u \cdot \nabla \delta u \mathrm{d}x + \lambda\int \delta u \mathrm{d}x - \int_\Omega f \delta u ~\mathrm{d}x - \int_{\partial \Omega} g \delta u~\mathrm{d}s = 0, \\
# \frac{\partial \mathcal{L}}{\partial \lambda}[\delta \lambda] &=\delta \lambda (\int_\Omega u \mathrm{d}x -h)= 0.
# \end{align}
# We write the weak formulation:
#
# $$
# \begin{align}
# \int_\Omega \nabla u \cdot \nabla \delta u~\mathrm{d}x + \int_\Omega \lambda \delta u~\mathrm{d}x = \int_\Omega f \delta u~\mathrm{d}x + \int_{\partial \Omega} g v \mathrm{d}s\\
# \int_\Omega u \delta \lambda  \mathrm{d}x = h \delta \lambda .
# \end{align}
# $$
#
# where we have moved $\delta\lambda$ into the integral as it is a spatial constant.

# ## Implementation
# We start by creating the domain and derive the source terms $f$, $g$ and $h$ from our manufactured solution
# For this example we will use the following exact solution
# \begin{align}
# u_{exact}(x, y) = 0.3y^2 + \sin(2\pi x).
# \end{align}

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.cpp.la.petsc import scatter_local_vectors, get_local_vectors
import dolfinx.fem.petsc

import numpy as np
from scifem import create_real_functionspace, assemble_scalar
import ufl

M = 20
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, M, M, dolfinx.mesh.CellType.triangle, dtype=np.float64
)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def u_exact(x):
    return 0.3 * x[1] ** 2 + ufl.sin(2 * ufl.pi * x[0])


x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)
g = ufl.dot(ufl.grad(u_exact(x)), n)
f = -ufl.div(ufl.grad(u_exact(x)))
h = assemble_scalar(u_exact(x) * ufl.dx)

# We then create the Lagrange multiplier space

R = create_real_functionspace(mesh)

# Next, we can create a mixed-function space for our problem

if dolfinx.__version__ == "0.8.0":
    u = ufl.TrialFunction(V)
    lmbda = ufl.TrialFunction(R)
    du = ufl.TestFunction(V)
    dl = ufl.TestFunction(R)
elif dolfinx.__version__ == "0.9.0.0":
    W = ufl.MixedFunctionSpace(V, R)
    u, lmbda = ufl.TrialFunctions(W)
    du, dl = ufl.TestFunctions(W)
else:
    raise RuntimeError("Unsupported version of dolfinx")

# We can now define the variational problem

zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

a00 = ufl.inner(ufl.grad(u), ufl.grad(du)) * ufl.dx
a01 = ufl.inner(lmbda, du) * ufl.dx
a10 = ufl.inner(u, dl) * ufl.dx
L0 = ufl.inner(f, du) * ufl.dx + ufl.inner(g, du) * ufl.ds
L1 = ufl.inner(zero, dl) * ufl.dx

a = dolfinx.fem.form([[a00, a01], [a10, None]])
L = dolfinx.fem.form([L0, L1])

# Note that we have defined the variational form in a block form, and
# that we have not included $h$ in the variational form. We will enforce this
# once we have assembled the right hand side vector.

# We can now assemble the matrix and vector

A = dolfinx.fem.petsc.assemble_matrix_block(a)
A.assemble()
b = dolfinx.fem.petsc.assemble_vector_block(L, a, bcs=[])

# Next, we modify the second part of the block to contain `h`
# We start by enforcing the multiplier constraint $h$ by modifying the right hand side vector

if dolfinx.__version__ == "0.8.0":
    maps = [(V.dofmap.index_map, V.dofmap.index_map_bs), (R.dofmap.index_map, R.dofmap.index_map_bs)]
elif dolfinx.__version__ == "0.9.0.0":
    maps = [(Wi.dofmap.index_map, Wi.dofmap.index_map_bs) for Wi in W.ufl_sub_spaces()]

b_local = get_local_vectors(b, maps)
b_local[1][:] = h
scatter_local_vectors(
        b,
        b_local,
        maps,
    )
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# We can now solve the linear system

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

xh = dolfinx.fem.petsc.create_vector_block(L)
ksp.solve(b, xh)
xh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# Finally, we extract the solution u from the blocked system and compute the error

uh = dolfinx.fem.Function(V, name="u")
x_local = get_local_vectors(xh, maps)
uh.x.array[: len(x_local[0])] = x_local[0]
uh.x.scatter_forward()


diff = uh - u_exact(x)
error = dolfinx.fem.form(ufl.inner(diff, diff) * ufl.dx)

print(f"L2 error: {np.sqrt(assemble_scalar(error)):.2e}")

# We can now plot the solution

vtk_mesh = dolfinx.plot.vtk_mesh(V)

import pyvista

pyvista.start_xvfb()
grid = pyvista.UnstructuredGrid(*vtk_mesh)
grid.point_data["u"] = uh.x.array.real

warped = grid.warp_by_scalar("u", factor=1)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe")
plotter.add_mesh(warped)
if not pyvista.OFF_SCREEN:
    plotter.show()
