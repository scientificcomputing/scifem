from mpi4py import MPI
import numpy as np
import dolfinx
import ufl


mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
cells = np.full(num_cells_local, 1, dtype=np.int32)

def some_cells(x):
    return x[0]<=x[1] 

cells[dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, some_cells)] = 2

submesh2, emap2 = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim, np.flatnonzero(cells==2))[:2]

C2_sub = dolfinx.fem.functionspace(submesh2, ("Lagrange", 1))
c2 = dolfinx.fem.Function(C2_sub)


V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

dx_sub = ufl.Measure("dx", domain=submesh2)
a = c2 * ufl.inner(u, v) * dx_sub

general_form = dolfinx.fem.compile_form(MPI.COMM_WORLD, a)

submesh1, emap1 = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim, np.flatnonzero(cells==1))[:2]
C1_sub = dolfinx.fem.functionspace(submesh1, ("Lagrange", 1))
c1_sub = dolfinx.fem.Function(C1_sub)

a1_sub = dolfinx.fem.create_form(general_form, function_spaces=[V, V], msh=submesh1, subdomains={}, coefficient_map={c2:c1_sub}, constant_map={}, entity_maps={emap1})
breakpoint()




