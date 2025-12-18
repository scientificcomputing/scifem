from mpi4py import MPI
import numpy as np
import dolfinx
import ufl


mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local + mesh.topology.index_map(mesh.topology.dim).num_ghosts
cells = np.full(num_cells_local, 1, dtype=np.int32)
ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), cells)

def some_cells(x):
    return x[0]<=x[1] 

def other_cells(x):
    return x[0] < 0.5

cells[dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, some_cells)] = 2
cells[dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, other_cells)] = 3

submesh2, emap2 = dolfinx.mesh.create_submesh(mesh, ct.dim, ct.find(2))[:2]

def f(x):
    return x[0] -x[1]**2

coeff_el = ("Lagrange", 2)
C2_sub = dolfinx.fem.functionspace(submesh2, coeff_el)
c2 = dolfinx.fem.Function(C2_sub)
c2.interpolate(f)


V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

dx_sub = ufl.Measure("dx", domain=submesh2)
a = c2 * ufl.inner(u, v) * dx_sub

general_form = dolfinx.fem.compile_form(MPI.COMM_WORLD, a)

# Create matrix for full matrix
a_matrix_alloc = dolfinx.fem.create_form(general_form, function_spaces=[V, V], msh=mesh, subdomains={}, coefficient_map={c2:c2}, constant_map={}, entity_maps=[emap2])
import dolfinx.fem.petsc
A = dolfinx.fem.petsc.create_matrix(a_matrix_alloc)
a2_sub = dolfinx.fem.create_form(general_form, function_spaces=[V, V], msh=submesh2, subdomains={}, coefficient_map={c2:c2}, constant_map={}, entity_maps={emap2})
dolfinx.fem.petsc.assemble_matrix(A, a2_sub)


for tag in [1, 3]:
    submesh, emap = dolfinx.mesh.create_submesh(mesh, ct.dim, ct.find(tag))[:2]
    C_sub = dolfinx.fem.functionspace(submesh, coeff_el)
    c_sub = dolfinx.fem.Function(C_sub)
    c_sub.interpolate(f)
    a_sub = dolfinx.fem.create_form(general_form, function_spaces=[V, V], msh=submesh, subdomains={}, coefficient_map={c2:c_sub}, constant_map={}, entity_maps={emap})
    dolfinx.fem.petsc.assemble_matrix(A, a_sub)
    del c_sub, C_sub, emap, submesh
A.assemble()

Q = dolfinx.fem.functionspace(mesh, coeff_el)
c = dolfinx.fem.Function(Q)
c.interpolate(f)
a_ref = c * ufl.inner(u, v) * ufl.dx
A_ref = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_ref))
A_ref.assemble()

Ar, Ac, Av = A.getValuesCSR()
Ar_ref, Ac_ref, Av_ref = A_ref.getValuesCSR()

np.testing.assert_allclose(Ar, Ar_ref)
np.testing.assert_allclose(Ac, Ac_ref)
np.testing.assert_allclose(Av, Av_ref)