from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
import scifem
import pytest


@pytest.mark.parametrize("dtype", [PETSc.RealType])
@pytest.mark.parametrize("tensor", [0, 1, 2])
@pytest.mark.parametrize("degree", range(1, 5))
def test_mixed_poisson(tensor, degree, dtype):
    M = 25
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, M, M, dolfinx.mesh.CellType.triangle, dtype=dtype
    )

    if tensor == 0:
        value_shape = ()
    elif tensor == 1:
        value_shape = (2,)
    else:
        value_shape = (3, 2)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, value_shape))
    R = scifem.create_real_functionspace(mesh, value_shape)

    W = ufl.MixedFunctionSpace(V, R)
    u, c = ufl.TrialFunctions(W)
    v, d = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(mesh)
    pol = x[0] ** degree - 2 * x[1] ** degree
    # Compute average value of polynomial to make mean 0
    C = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(pol * ufl.dx, dtype=dtype)), op=MPI.SUM
    )
    u_scalar = pol - dolfinx.fem.Constant(mesh, dtype(C))
    if tensor == 0:
        u_ex = u_scalar
        zero = dolfinx.fem.Constant(mesh, dtype(0.0))
    elif tensor == 1:
        u_ex = ufl.as_vector([u_scalar, -u_scalar])
        zero = dolfinx.fem.Constant(mesh, dtype((0.0, 0.0)))
    else:
        u_ex = ufl.as_tensor(
            [
                [u_scalar, 2 * u_scalar],
                [3 * u_scalar, -u_scalar],
                [u_scalar, 2 * u_scalar],
            ]
        )
        zero = dolfinx.fem.Constant(mesh, dtype(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))))

    dx = ufl.Measure("dx", domain=mesh)
    f = -ufl.div(ufl.grad(u_ex))
    n = ufl.FacetNormal(mesh)
    g = ufl.dot(ufl.grad(u_ex), n)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a += ufl.inner(c, v) * dx
    a += ufl.inner(u, d) * dx
    L = ufl.inner(f, v) * dx + ufl.inner(g, v) * ufl.ds
    L += ufl.inner(zero, d) * dx

    a_blocked = ufl.extract_blocks(a)
    L_blocked = ufl.extract_blocks(L)

    a = dolfinx.fem.form(a_blocked, dtype=dtype)
    L = dolfinx.fem.form(L_blocked, dtype=dtype)

    A = dolfinx.fem.petsc.assemble_matrix_block(a)
    A.assemble()
    b = dolfinx.fem.petsc.create_vector_block(L)
    with b.localForm() as loc:
        loc.set(0)
    dolfinx.fem.petsc.assemble_vector_block(b, L, a, bcs=[])

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    x = dolfinx.fem.petsc.create_vector_block(L)
    ksp.solve(b, x)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    uh = dolfinx.fem.Function(V)
    x_local = dolfinx.cpp.la.petsc.get_local_vectors(
        x,
        [(V.dofmap.index_map, V.dofmap.index_map_bs), (R.dofmap.index_map, R.dofmap.index_map_bs)],
    )
    uh.x.array[: len(x_local[0])] = x_local[0]
    uh.x.scatter_forward()

    error = dolfinx.fem.form(ufl.inner(u_ex - uh, u_ex - uh) * dx, dtype=dtype)

    e_local = dolfinx.fem.assemble_scalar(error)
    tol = 2e3 * np.finfo(dtype).eps
    e_global = np.sqrt(mesh.comm.allreduce(e_local, op=MPI.SUM))
    assert np.isclose(e_global, 0, atol=tol)

