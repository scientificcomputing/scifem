from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import dolfinx
import scifem
import ufl

M = 250
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, M, M)

V = dolfinx.fem.functionspace(mesh, ("N1curl", 1))


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
k = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.1))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
a += 1j * ufl.inner(u, v) * ufl.ds


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

geom_dtype = mesh.geometry.x.dtype
if mesh.comm.rank == 0:
    points = np.array([[0.68, 0.362, 0], [0.14, 0.213, 0]], dtype=geom_dtype)
else:
    points = np.zeros((0, 3), dtype=geom_dtype)

# Next, we create the point source object and apply it to the right hand side vector.

gamma = 1.0
point_source = scifem.PointSource(V, points, magnitude=gamma)
point_source.apply_to_vector(b)

# We can continue to solve our variational problem as usual

bcs = []
a_compiled = dolfinx.fem.form(a)
dolfinx.fem.petsc.apply_lifting(b.x.petsc_vec, [a_compiled], [bcs])
b.x.scatter_reverse(dolfinx.la.InsertMode.add)
dolfinx.fem.petsc.set_bc(b.x.petsc_vec, bcs)
b.x.scatter_forward()

A = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=bcs)
A.assemble()

# Set up a direct solver an solve the linear system

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.setErrorIfNotConverged(True)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.getPC().setFactorSolverType("mumps")


uh = dolfinx.fem.Function(V)
ksp.solve(b.x.petsc_vec, uh.x.petsc_vec)
uh.x.scatter_forward()


V_out = dolfinx.fem.functionspace(mesh, ("DG", 1, (2,)))
u_out = dolfinx.fem.Function(V_out)
u_out.interpolate(uh)

with dolfinx.io.VTXWriter(mesh.comm, "point_source.bp", [u_out]) as bp:
    bp.write(0.0)
