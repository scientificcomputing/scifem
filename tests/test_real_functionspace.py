from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
import scifem
import pytest


@pytest.mark.parametrize("L", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("H", [1.3, 0.8, 0.2])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_real_function_space_mass(L, H, cell_type, dtype):
    """
    Check that real space mass matrix is the same as assembling the volume of the mesh
    """

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [7, 9], cell_type, dtype=dtype
    )

    V = scifem.create_real_functionspace(mesh)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx

    A = dolfinx.fem.assemble_matrix(dolfinx.fem.form(a, dtype=dtype), bcs=[])
    A.scatter_reverse()
    tol = 100 * np.finfo(dtype).eps

    assert len(A.data) == 1
    if MPI.COMM_WORLD.rank == 0:
        assert np.isclose(A.data[0], L * H, atol=tol)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
def test_real_function_space_vector(cell_type, dtype):
    """
    Test that assembling against a real space test function is equivalent to assembling a vector
    """

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 3, 5, cell_type, dtype=dtype)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))
    v = ufl.TrialFunction(V)

    R = scifem.create_real_functionspace(mesh)
    u = ufl.TestFunction(R)
    a_R = ufl.inner(u, v) * ufl.dx
    form_rhs = dolfinx.fem.form(a_R, dtype=dtype)

    A_R = dolfinx.fem.assemble_matrix(form_rhs, bcs=[])
    A_R.scatter_reverse()

    L = ufl.inner(ufl.constantvalue.IntValue(1), v) * ufl.dx
    form_lhs = dolfinx.fem.form(L, dtype=dtype)
    b = dolfinx.fem.assemble_vector(form_lhs)
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    b.scatter_forward()

    row_map = A_R.index_map(0)
    num_local_rows = row_map.size_local
    num_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    tol = 100 * np.finfo(dtype).eps
    if MPI.COMM_WORLD.rank == 0:
        assert num_local_rows == 1
        num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        assert A_R.indptr[1] - A_R.indptr[0] == num_dofs_global
        np.testing.assert_allclose(A_R.indices, np.arange(num_dofs_global))
        np.testing.assert_allclose(b.array[:num_dofs], A_R.data[:num_dofs], atol=tol)
    else:
        assert num_local_rows == 0


@pytest.mark.parametrize("dtype", [(PETSc.RealType, PETSc.ScalarType)])
@pytest.mark.parametrize("tensor", [0, 1, 2])
@pytest.mark.parametrize("degree", range(1, 5))
def test_singular_poisson(tensor, degree, dtype):
    M = 25

    rtype, stype = dtype
    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, M, M, dolfinx.mesh.CellType.triangle, dtype=rtype
    )

    if tensor == 0:
        value_shape = ()
    elif tensor == 1:
        value_shape = (2,)
    else:
        value_shape = (3, 2)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, value_shape))
    R = scifem.create_real_functionspace(mesh, value_shape)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    c = ufl.TrialFunction(R)
    d = ufl.TestFunction(R)
    x = ufl.SpatialCoordinate(mesh)
    pol = x[0] ** degree - 2 * x[1] ** degree
    # Compute average value of polynomial to make mean 0
    C = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(pol * ufl.dx, dtype=stype)), op=MPI.SUM
    )
    u_scalar = pol - dolfinx.fem.Constant(mesh, stype(C))
    if tensor == 0:
        u_ex = u_scalar
        zero = dolfinx.fem.Constant(mesh, stype(0.0))
    elif tensor == 1:
        u_ex = ufl.as_vector([u_scalar, -u_scalar])
        zero = dolfinx.fem.Constant(mesh, stype((0.0, 0.0)))
    else:
        u_ex = ufl.as_tensor(
            [
                [u_scalar, 2 * u_scalar],
                [3 * u_scalar, -u_scalar],
                [u_scalar, 2 * u_scalar],
            ]
        )
        zero = dolfinx.fem.Constant(mesh, stype(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))))

    dx = ufl.Measure("dx", domain=mesh)
    f = -ufl.div(ufl.grad(u_ex))
    n = ufl.FacetNormal(mesh)
    g = ufl.dot(ufl.grad(u_ex), n)
    a00 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a01 = ufl.inner(c, v) * dx
    a10 = ufl.inner(u, d) * dx

    L0 = ufl.inner(f, v) * dx + ufl.inner(g, v) * ufl.ds
    L1 = ufl.inner(zero, d) * dx

    a = dolfinx.fem.form([[a00, a01], [a10, None]], dtype=stype)
    L = dolfinx.fem.form([L0, L1], dtype=stype)

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

    error = dolfinx.fem.form(ufl.inner(u_ex - uh, u_ex - uh) * dx, dtype=stype)

    e_local = dolfinx.fem.assemble_scalar(error)
    tol = 1e4 * np.finfo(stype).eps
    e_global = np.sqrt(mesh.comm.allreduce(e_local, op=MPI.SUM))
    assert np.isclose(e_global, 0, atol=tol)


@pytest.mark.parametrize("ftype, stype", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_complex_real_space(ftype, stype):
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 13, dtype=ftype)

    val = (2 + 3j, -4 + 5j)
    value_shape = (2,)
    R = scifem.create_real_functionspace(mesh, value_shape=value_shape)
    u = dolfinx.fem.Function(R, dtype=stype)
    u.x.array[0] = val[0]
    u.x.array[1] = val[1]
    u.x.scatter_forward()

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, value_shape))
    v = ufl.TestFunction(V)
    L = ufl.inner(u, v) * ufl.dx

    b = dolfinx.fem.assemble_vector(dolfinx.fem.form(L, dtype=stype))
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    b.scatter_forward()
    const = dolfinx.fem.Constant(mesh, stype(val))
    L_const = ufl.inner(const, v) * ufl.dx
    b_const = dolfinx.fem.assemble_vector(dolfinx.fem.form(L_const, dtype=stype))
    b_const.scatter_reverse(dolfinx.la.InsertMode.add)
    b_const.scatter_forward()

    tol = 100 * np.finfo(stype).eps
    np.testing.assert_allclose(b.array, b_const.array, atol=tol)
