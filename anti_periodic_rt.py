from script import create_periodic_mesh
from mpi4py import MPI
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import dolfinx
import numpy as np
import ufl
from petsc4py import PETSc


mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0, 0], [2, 1]],
    [50, 7],
    dolfinx.cpp.mesh.CellType.quadrilateral,
)

# Convert mesh to periodic mesh
L_min = [
    mesh.comm.allreduce(np.min(mesh.geometry.x[:, i]), op=MPI.MIN) for i in range(2)
]
L_max = [
    mesh.comm.allreduce(np.max(mesh.geometry.x[:, i]), op=MPI.MAX) for i in range(2)
]

print(L_min, L_max)


def i_x(x):
    return np.isclose(x[0], L_min[0])


def indicator(x):
    return i_x(x)


def mapping(x):
    values = x.copy()
    values[0] = L_min[0] + i_x(x) * (L_max[0] - L_min[0])
    values[1] = L_max[1] - (L_max[1] - L_min[1]) * x[1]
    print(values[0], values[1])
    return values


mesh.topology.create_entities(mesh.topology.dim - 1)

print(MPI.COMM_WORLD.rank, f"map x from {L_min} to {L_max}")
mesh.topology.create_entities(mesh.topology.dim - 1)
print(mesh.topology.index_map(mesh.topology.dim - 1).size_local)

mesh, replaced_vertices, replacement_map = create_periodic_mesh(
    mesh, indicator, mapping
)

mesh.topology.create_entities(mesh.topology.dim - 1)
print(mesh.topology.index_map(mesh.topology.dim - 1).size_local)
import basix.ufl

el_0 = basix.ufl.element("DG", mesh.topology.cell_name(), 2)
el_1 = basix.ufl.element("RT", mesh.topology.cell_name(), 3)
trial_el = basix.ufl.mixed_element([el_0, el_1])
V = dolfinx.fem.functionspace(mesh, trial_el)

w = dolfinx.fem.Function(V)
u, psi = ufl.split(w)

v, tau = ufl.TestFunctions(V)

dx = ufl.Measure("dx", domain=mesh)

uD = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))
U, U_to_W = V.sub(0).collapse()
Q, Q_to_W = V.sub(1).collapse()
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))


alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
phi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
w0 = dolfinx.fem.Function(V)
u0, psi0 = ufl.split(w0)

F = ufl.inner(ufl.div(psi), v) * dx
F -= ufl.inner(ufl.div(psi0), v) * dx
F += alpha * ufl.inner(f, v) * dx

non_lin_term = 1 / (ufl.sqrt(1 + ufl.dot(psi, psi)))
F += ufl.inner(u, ufl.div(tau)) * dx
F += phi * non_lin_term * ufl.dot(psi, tau) * dx


J = ufl.derivative(F, w)

tol = 1e-5

problem = dolfinx.fem.petsc.NonlinearProblem(F, w, bcs=[], J=J)
solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
solver.convergence_criterion = "residual"
solver.rtol = tol
solver.atol = tol
solver.max_it = 100
solver.error_on_nonconvergence = True


ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
V_out = dolfinx.fem.functionspace(mesh, ("DG", mesh.geometry.cmap.degree))
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
    nh, Q.element.interpolation_points(), dtype=dolfinx.default_scalar_type
)
f_to_c = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
c_to_f = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 1)
ie = []
for facet in entity_map:
    cells = f_to_c.links(facet)
    if len(cells) > 1:
        cell = f_to_c.links(facet)[1]
    else:
        cell = f_to_c.links(facet)[0]
    facets = c_to_f.links(cell)
    local_index = np.flatnonzero(facets == facet)[0]
    ie.append(cell)
    ie.append(local_index)
values = expr.eval(mesh, np.asarray(ie, dtype=np.int32))
qq = dolfinx.fem.Function(Q)
qq.x.array[:] = values.flatten()

try:
    newton_iterations = []
    for i in range(1, 100):
        alpha.value = min(2**i, 10)

        num_newton_iterations, converged = solver.solve(w)
        newton_iterations.append(num_newton_iterations)
        print(
            f"Iteration {i}: {converged=} {num_newton_iterations=} {ksp.getConvergedReason()=}"
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
