from mpi4py import MPI

import numpy as np
import dolfinx
from petsc4py import PETSc
from scifem import NewtonSolver, assemble_scalar
import ufl
import pytest


@pytest.mark.parametrize("factor", [1, -1])
def test_NewtonSolver(factor):
    dtype = PETSc.RealType
    ftype = PETSc.ScalarType
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 12, dtype=dtype)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))

    # if dolfinx.__version__ == "0.8.0":
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)
    du = ufl.TrialFunction(V)
    dp = ufl.TrialFunction(Q)
    # elif dolfinx.__version__ == "0.9.0.0":
    # Relies on: https://github.com/FEniCS/ufl/pull/308
    # W = ufl.MixedFunctionSpace(V, Q)
    # v, q = ufl.TestFunctions(W)
    # du, dp = ufl.TrialFunctions(W)
    # else:
    #     raise ValueError("Unsupported version of dolfinx")

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
    # if dolfinx.__version__ == "0.8.0":
    F = [F0, F1]
    J = [
        [ufl.derivative(F0, u, du), ufl.derivative(F0, p, dp)],
        [ufl.derivative(F1, u, du), ufl.derivative(F1, p, dp)],
    ]
    # elif dolfinx.__version__ == "0.9.0.0":
    # Relies on: https://github.com/FEniCS/ufl/pull/308
    # F_ = F0 + F1
    # F = list(ufl.extract_blocks(F_))
    # J = ufl.extract_blocks(ufl.derivative(F_, u, du) + ufl.derivative(F_, p, dp))
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    solver = NewtonSolver(F, J, [u, p], max_iterations=25, petsc_options=petsc_options)
    solver.solve()

    u_ex = factor * ufl.sqrt(u_expr / c0)
    p_ex = factor * ufl.sqrt(p_expr / c1)
    err_u = ufl.inner(u_ex - u, u_ex - u) * ufl.dx
    err_p = ufl.inner(p_ex - p, p_ex - p) * ufl.dx
    tol = np.finfo(dtype).eps * 1.0e3
    assert assemble_scalar(err_u) < tol
    assert assemble_scalar(err_p) < tol
