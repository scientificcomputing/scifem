from mpi4py import MPI

import numpy as np
import dolfinx
import dolfinx.nls.petsc
from petsc4py import PETSc
from scifem import assemble_scalar, BlockedNewtonSolver, NewtonSolver
import basix.ufl
import ufl
import pytest


@pytest.mark.parametrize("factor", [1, -1])
@pytest.mark.parametrize("auto_split", [True, False])
def test_NewtonSolver(factor, auto_split):
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    dtype = PETSc.RealType
    ftype = PETSc.ScalarType
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 12, dtype=dtype)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, left_boundary)

    el_0 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    el_1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2)
    V = dolfinx.fem.functionspace(mesh, el_0)
    Q = dolfinx.fem.functionspace(mesh, el_1)

    if auto_split:
        W = ufl.MixedFunctionSpace(V, Q)
        v, q = ufl.TestFunctions(W)
        du, dp = ufl.TrialFunctions(W)
    else:
        v = ufl.TestFunction(V)
        q = ufl.TestFunction(Q)
        du = ufl.TrialFunction(V)
        dp = ufl.TrialFunction(Q)

    # Set nonzero initial guess
    # Choose initial guess acording to the input factor
    u = dolfinx.fem.Function(V, dtype=ftype)
    u.x.array[:] = factor * 0.1
    p = dolfinx.fem.Function(Q, dtype=ftype)
    p.x.array[:] = factor * 0.02
    x = ufl.SpatialCoordinate(mesh)
    u_expr = 3 * x[0] ** 2
    p_expr = 5 * x[1] ** 4
    c0 = dolfinx.fem.Constant(mesh, ftype(0.3))
    c1 = dolfinx.fem.Constant(mesh, ftype(0.82))
    F0 = ufl.inner(c0 * u**2, v) * ufl.dx - ufl.inner(u_expr, v) * ufl.dx
    F1 = ufl.inner(c1 * p**2, q) * ufl.dx - ufl.inner(p_expr, q) * ufl.dx
    if auto_split:
        F_ = F0 + F1
        F = list(ufl.extract_blocks(F_))
        J = ufl.extract_blocks(ufl.derivative(F_, u, du) + ufl.derivative(F_, p, dp))
    else:
        F = [F0, F1]
        J = [
            [ufl.derivative(F0, u, du), ufl.derivative(F0, p, dp)],
            [ufl.derivative(F1, u, du), ufl.derivative(F1, p, dp)],
        ]

    # Reference solution
    u_ex = factor * ufl.sqrt(u_expr / c0)
    p_ex = factor * ufl.sqrt(p_expr / c1)

    # Create BC on second space
    dofs_q = dolfinx.fem.locate_dofs_topological(Q, mesh.topology.dim - 1, boundary_facets)
    p_bc = dolfinx.fem.Function(Q, dtype=ftype)
    try:
        ip = Q.element.interpolation_points()

    except TypeError:
        ip = Q.element.interpolation_points
    p_bc.interpolate(dolfinx.fem.Expression(p_ex, ip))
    bc_q = dolfinx.fem.dirichletbc(p_bc, dofs_q)

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    solver = NewtonSolver(F, J, [u, p], bcs=[bc_q], max_iterations=25, petsc_options=petsc_options)
    solver.solve()

    err_u = ufl.inner(u_ex - u, u_ex - u) * ufl.dx
    err_p = ufl.inner(p_ex - p, p_ex - p) * ufl.dx
    tol = np.finfo(dtype).eps * 1.0e3
    assert assemble_scalar(err_u) < tol
    assert assemble_scalar(err_p) < tol

    # Check consistency with other Newton solver
    blocked_solver = BlockedNewtonSolver(
        F, [u, p], bcs=[bc_q], J=None if auto_split else J, petsc_options=petsc_options
    )

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    u.x.array[:] = factor * 0.1
    p.x.array[:] = factor * 0.02
    blocked_solver.convergence_criterion = "incremental"
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    blocked_solver.solve()

    err_u = ufl.inner(u_ex - u, u_ex - u) * ufl.dx
    err_p = ufl.inner(p_ex - p, p_ex - p) * ufl.dx
    tol = np.finfo(dtype).eps * 1.0e3
    assert assemble_scalar(err_u) < tol
    assert assemble_scalar(err_p) < tol
