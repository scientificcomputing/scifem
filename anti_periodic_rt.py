from script import create_periodic_mesh, transfer_meshtags_to_periodic_mesh
from mpi4py import MPI
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import dolfinx
import numpy as np
import ufl
from petsc4py import PETSc


_mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0, 0.1], [2, 1]],
    [50, 25],
    dolfinx.cpp.mesh.CellType.quadrilateral,
)

# Convert mesh to periodic mesh
L_min = [
    _mesh.comm.allreduce(np.min(_mesh.geometry.x[:, i]), op=MPI.MIN) for i in range(2)
]
L_max = [
    _mesh.comm.allreduce(np.max(_mesh.geometry.x[:, i]), op=MPI.MAX) for i in range(2)
]

print(L_min, L_max)


def i_x(x):
    return np.isclose(x[0], L_min[0])


def indicator(x):
    return i_x(x)


def mapping(x):
    values = x.copy()
    values[0] = L_min[0] + i_x(x) * (L_max[0] - L_min[0])
    # Comment out for normal periodic version
    values[1] = (L_max[1] - L_min[1]) / (L_min[1] - L_max[1]) * x[1] + (
        L_min[1] ** 2 - L_max[1] ** 2
    ) / (L_min[1] - L_max[1])
    return values


_mesh.topology.create_entities(_mesh.topology.dim - 1)

print(MPI.COMM_WORLD.rank, f"map x from {L_min} to {L_max}")
_mesh.topology.create_entities(_mesh.topology.dim - 1)
print(_mesh.topology.index_map(_mesh.topology.dim - 1).size_local)

num_facets = (
    _mesh.topology.index_map(_mesh.topology.dim - 1).size_local
    + _mesh.topology.index_map(_mesh.topology.dim - 1).num_ghosts
)
marker = np.full(num_facets, 3, dtype=np.int32)
_mesh.topology.create_connectivity(_mesh.topology.dim - 1, _mesh.topology.dim)
marker[dolfinx.mesh.exterior_facet_indices(_mesh.topology)] = 1
_ft = dolfinx.mesh.meshtags(
    _mesh, _mesh.topology.dim - 1, np.arange(num_facets, dtype=np.int32), marker
)
mesh, replaced_vertices, replacement_map = create_periodic_mesh(
    _mesh, indicator, mapping
)
ft = transfer_meshtags_to_periodic_mesh(_mesh, mesh, replaced_vertices, _ft)

# with dolfinx.io.XDMFFile(mesh.comm, "facets.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
#     xdmf.write_meshtags(ft, mesh.geometry)

mesh.topology.create_entities(mesh.topology.dim - 1)
print(mesh.topology.index_map(mesh.topology.dim - 1).size_local)
import basix.ufl

el_0 = basix.ufl.element("DG", mesh.topology.cell_name(), 1)
el_1 = basix.ufl.element("RT", mesh.topology.cell_name(), 2)
trial_el = basix.ufl.mixed_element([el_0, el_1])
V = dolfinx.fem.functionspace(mesh, trial_el)
w = dolfinx.fem.Function(V)
u, psi = ufl.split(w)

v, tau = ufl.TestFunctions(V)

dx = ufl.Measure("dx", domain=mesh)

uD = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))
U, U_to_W = V.sub(0).collapse()
Q, Q_to_W = V.sub(1).collapse()
x = ufl.SpatialCoordinate(mesh)
f = x[0] + x[1]*dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))


alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
phi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
w0 = dolfinx.fem.Function(V)
u0, psi0 = ufl.split(w0)

F = ufl.inner(ufl.div(psi), v) * dx
F -= ufl.inner(ufl.div(psi0), v) * dx
F += alpha * ufl.inner(f, v) * dx
F += ufl.inner(u, ufl.div(tau)) * dx

non_lin_term = 1 / (ufl.sqrt(1 + ufl.dot(psi, psi)))
F += phi * non_lin_term * ufl.dot(psi, tau) * dx


J = ufl.derivative(F, w)

tol = 1e-5

problem = dolfinx.fem.petsc.NonlinearProblem(F, u=w, bcs=[], J=J,
                                             petsc_options_prefix="rt_",
                                             petsc_options={
                                                 "ksp_type": "preonly",
                                                 "pc_type": "lu",
                                                 "pc_factor_mat_solver_type": "mumps",
                                                 "ksp_error_if_not_converged": True,
                                                 "ksp_monitor": None,
                                                 "snes_type": "newtonls",
                                                 "snes_linesearch_type": "basic",
                                                 "snes_rtol": 1e-10,
                                                 "snes_atol": 1e-10,
                                                 "snes_monitor": None,
                                                 "snes_error_if_not_converged": True,
                                                 "snes_max_it": 100,})




dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
V_out = dolfinx.fem.functionspace(mesh, ("DG", 2))
u_out = dolfinx.fem.Function(V_out)
u_out.name = "u"
bp_u = dolfinx.io.VTXWriter(mesh.comm, "u_rt.bp", [u_out])
diff = w.sub(0) - w0.sub(0)
L2_squared = ufl.dot(diff, diff) * dx
compiled_diff = dolfinx.fem.form(L2_squared)


nh = ufl.FacetNormal(mesh)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
num_facets = mesh.topology.index_map(mesh.topology.dim - 1).size_local
submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim - 1, np.arange(num_facets, dtype=np.int32)
)
q_el = basix.ufl.quadrature_element(submesh.basix_cell(), nh.ufl_shape, "default", 1)
Q = dolfinx.fem.functionspace(submesh, q_el)
expr = dolfinx.fem.Expression(
    nh, Q.element.interpolation_points, dtype=dolfinx.default_scalar_type
)
f_to_c = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
c_to_f = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 1)
ie = []

sub_cell_map = submesh.topology.index_map(submesh.topology.dim)
num_sub_cells = sub_cell_map.size_local + sub_cell_map.num_ghosts
parent_facets = entity_map.sub_topology_to_topology(np.arange(num_sub_cells, dtype=np.int32), inverse=False)
for facet in parent_facets:
    cells = f_to_c.links(facet)
    if len(cells) > 1:
        cell = f_to_c.links(facet)[1]
    else:
        cell = f_to_c.links(facet)[0]
    facets = c_to_f.links(cell)
    local_index = np.flatnonzero(facets == facet)[0]
    ie.append(cell)
    ie.append(local_index)
values = expr.eval(mesh, np.asarray(ie, dtype=np.int32).reshape((-1, 2)))
qq = dolfinx.fem.Function(Q)
qq.x.array[:] = values.flatten()

try:
    newton_iterations = []
    for i in range(1, 100):
        alpha.value = min(2**i, 10)

        problem.solve()
        num_newton_iterations = problem.solver.getIterationNumber()
        newton_iterations.append(num_newton_iterations)
        print(
            f"Iteration {i}: {num_newton_iterations=} {problem.solver.getConvergedReason()=}"
        )
        local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
        global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
        print(f"|delta u |= {global_diff}")
        w0.x.array[:] = w.x.array

        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

        u_out.interpolate(w.sub(0))
        bp_u.write(i)

        if global_diff < 5 * tol:
            break
finally:
    bp_u.close()

print(
    f"Num LVPP iterations {i}, Total number of newton iterations {sum(newton_iterations)}"
)
print(f"{min(newton_iterations)=} and {max(newton_iterations)=}")
print(f"NUM DOFS: {V.dofmap.index_map.size_global * V.dofmap.bs}")
