"""Suitable mesh created with

python3 create_mesh.py --res=0.01 --periodic
"""

from mpi4py import MPI
import dolfinx
import numpy as np
from script import create_periodic_mesh, transfer_meshtags_to_periodic_mesh
import time
import basix.ufl
import ufl
import dolfinx.fem.petsc
import dolfinx.nls.petsc
if __name__ == "__main__":
    max_facet_to_cell_links=2
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet, max_facet_to_cell_links=max_facet_to_cell_links)
    mesh_data =  dolfinx.io.gmsh.read_from_msh("mesh.msh", MPI.COMM_WORLD,  0, 2, partitioner=partitioner)
    mesh = mesh_data.mesh
    ct = mesh_data.cell_tags
    ft = mesh_data.facet_tags

    L_min = MPI.COMM_WORLD.allreduce(np.min(mesh.geometry.x[:,0]), op=MPI.MIN)
    L_max = MPI.COMM_WORLD.allreduce(np.max(mesh.geometry.x[:,0]), op=MPI.MAX)


    def indicator(x):
        return np.isclose(x[0], L_min)

    def mapping(x):
        values = x.copy()
        values[0] += L_max-L_min
        return values


    start = time.perf_counter()
    new_mesh, replaced_vertices, replacement_map = create_periodic_mesh(mesh, indicator, mapping)
    end = time.perf_counter()
    print(f"Create periodic mesh: {end-start:.3e}")

    facet_tags = transfer_meshtags_to_periodic_mesh(mesh, new_mesh, replaced_vertices, ft)


    el_u = basix.ufl.element("Lagrange", new_mesh.basix_cell(), 2, shape=(new_mesh.geometry.dim, ))
    if mesh.topology.cell_type == dolfinx.cpp.mesh.CellType.triangle:
        el_p = basix.ufl.element("DG", new_mesh.basix_cell(), 0, shape=())
    else:
        el_p = basix.ufl.element("DPC", new_mesh.basix_cell(), 1, shape=())

    mixed_el = basix.ufl.mixed_element([el_u, el_p])

    W = dolfinx.fem.functionspace(new_mesh, mixed_el)
    w = dolfinx.fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)


    w_n = dolfinx.fem.Function(W)
    u_n, _ = ufl.split(w_n)
    
    dt = 5e-4
    k = dolfinx.fem.Constant(new_mesh, dolfinx.default_scalar_type(dt))
    mu = dolfinx.fem.Constant(new_mesh, dolfinx.default_scalar_type(0.01))  # Dynamic viscosity
    rho = dolfinx.fem.Constant(new_mesh,dolfinx.default_scalar_type(1))     
    du_dt = u - u_n
    F = rho*ufl.inner(du_dt, v) * ufl.dx
    F += k * mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += k * rho * ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
    F += k * ufl.div(v) * p * ufl.dx 
    F += ufl.div(u) * q * ufl.dx
    x = ufl.SpatialCoordinate(new_mesh)
    source = dolfinx.fem.Constant(new_mesh, dolfinx.default_scalar_type(0.3))
    F -= ufl.inner(source, v[0])*ufl.dx


    W0, sub0_to_mixed = W.sub(0).collapse()
    new_mesh.topology.create_connectivity(new_mesh.topology.dim - 1, new_mesh.topology.dim)
    wall_dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), W0), new_mesh.topology.dim-1, facet_tags.find(1))
    u_bc = dolfinx.fem.Function(W0)
    bc_wall = dolfinx.fem.dirichletbc(u_bc, wall_dofs, W.sub(0))
    bcs = [bc_wall]

    problem = dolfinx.fem.petsc.NonlinearProblem(F, u=w, bcs=bcs,petsc_options_prefix="ns_",
                                                 petsc_options={
                                                     "ksp_type": "preonly",
                                                     "pc_type": "lu",
                                                     "pc_factor_mat_solver_type": "mumps",
                                                     "mat_mumps_icntl_24": 1,
                                                     "ksp_error_if_not_converged": True,
                                                     #"ksp_monitor": None,
                                                     "snes_type": "newtonls",
                                                     "snes_linesearch_type": "basic",
                                                     "snes_rtol": 1e-10,
                                                     "snes_atol": 1e-10,
                                                     #"snes_monitor": None,
                                                     "snes_error_if_not_converged": True,})

    W1, sub1_to_mixed = W.sub(1).collapse()
    ns_vec = dolfinx.fem.Function(W)
    ns_vec.x.array[sub1_to_mixed] = 1
    dolfinx.la.orthonormalize([ns_vec.x])
    #nullspace = PETSc.NullSpace().create(vectors=[ns_vec.x.petsc_vec])

    #problem.A.setNullSpace(nullspace)
    t = 0
    T = 2000*dt
    num_steps = int(T/dt)
    W_out = dolfinx.fem.functionspace(new_mesh, ("DG", 2, (new_mesh.geometry.dim, )))
    u_out = dolfinx.fem.Function(W_out)
    interpolation_matrix_u = dolfinx.fem.petsc.interpolation_matrix(W.sub(0), W_out)
    interpolation_matrix_u.assemble()
    bp = dolfinx.io.VTXWriter(new_mesh.comm, "u.bp", [u_out])
    bp.write(0.0)
    
    
    Q_out = dolfinx.fem.functionspace(new_mesh, ("DG", 1))
    p_out = dolfinx.fem.Function(Q_out)
    bp_p = dolfinx.io.VTXWriter(new_mesh.comm, "p.bp", [p_out])
    interpolation_matrix_p = dolfinx.fem.petsc.interpolation_matrix(W.sub(1), Q_out)
    interpolation_matrix_p.assemble()

    interpolation_matrix_p.mult(w.x.petsc_vec, p_out.x.petsc_vec)
    p_out.x.scatter_forward()
    bp_p.write(0.0)
    stationary_counter = 0
    max_stationary = 10
    for i in range(num_steps):
        t += dt

        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        problem.solve()
        
        converged_reason = problem.solver.getConvergedReason()     
        w_n.x.array[:] = w.x.array

        interpolation_matrix_u.mult(w.x.petsc_vec, u_out.x.petsc_vec)
        u_out.x.scatter_forward()
        bp.write(t)


        interpolation_matrix_p.mult(w.x.petsc_vec, p_out.x.petsc_vec)
        p_out.x.scatter_forward()

        bp_p.write(t)
        num_its = problem.solver.getIterationNumber()
        print(f"Step {i+1}/{num_steps}, {t=:.2e}, {num_its=}, {converged_reason=}")
        if num_its == 0:
            stationary_counter += 1
            if stationary_counter >= max_stationary:
                print("Reached max stationary counter")
                break
        else:
            stationary_counter = 0
    bp.close()
    bp_p.close()