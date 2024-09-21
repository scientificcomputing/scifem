# # Point sources in DOLFINx
# 
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT

# In this example, we will illustrate how to apply a point source to a Poisson problem.
#
# \begin{align}
# -\nabla^2 u &= f \quad \text{in } \Omega, \\
# f &= \gamma \delta(x - p) \quad \text{in } \Omega,

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl
import numpy as np
import scifem

# We start by creating the mesh and function space
N = 40
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
domain.name = "mesh"
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim-1, tdim)
d = value_shape = (2,)
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, value_shape))

facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx


dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
u_bc = dolfinx.fem.Function(V)
u_bc.x.array[:] = 0
bc = dolfinx.fem.dirichletbc(u_bc, dofs)

b = dolfinx.fem.Function(V)
b.x.array[:] = 0

geom_dtype = domain.geometry.x.dtype
if domain.comm.rank == 0:
    points = np.array([[0.68, 0.362, 0],
                       [0.14, 0.213, 0]], dtype=geom_dtype)
else:
    points = np.zeros((0, 3), dtype=geom_dtype)

gamma = 1.
point_source = scifem.PointSource(V.sub(1), points, magnitude=gamma)
point_source.apply_to_vector(b)

a_compiled = dolfinx.fem.form(a)
dolfinx.fem.petsc.apply_lifting(b.vector, [a_compiled], [[bc]])
b.x.scatter_reverse(dolfinx.la.InsertMode.add)
dolfinx.fem.petsc.set_bc(b.vector, [bc])
b.x.scatter_forward()

A = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=[bc])
A.assemble()

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.getPC().setFactorSolverType("mumps")


uh = dolfinx.fem.Function(V)
ksp.solve(b.vector, uh.vector)
uh.x.scatter_forward()

with dolfinx.io.VTXWriter(domain.comm, "uh.bp", [uh], engine="BP5") as bp:
    bp.write(0.0)
