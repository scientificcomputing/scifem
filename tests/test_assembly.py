from mpi4py import MPI

from scifem import assemble_scalar, norm
import scifem.petsc
from dolfinx.mesh import create_unit_square, exterior_facet_indices
import dolfinx
import basix.ufl
from dolfinx.fem import (
    Function,
    functionspace,
    Constant,
    form,
    locate_dofs_topological,
    dirichletbc,
)
from dolfinx import default_scalar_type
import ufl
import pytest
import numpy as np


@pytest.mark.parametrize("gtype", [np.float64, np.float32])
def test_assemble_scalar_spatial(gtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=gtype)
    x = ufl.SpatialCoordinate(mesh)
    f = x[0] ** 2 * x[1] * ufl.dx
    tol = 50 * np.finfo(gtype).eps
    assert np.isclose(assemble_scalar(f), 1 / 3 * 1 / 2, atol=tol)


@pytest.mark.parametrize(
    "gtype, dtype",
    [
        [np.float64, np.float64],
        [np.float64, np.complex128],
        [np.float32, np.float32],
        [np.float32, np.complex64],
    ],
)
def test_assemble_scalar_constant(gtype, dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=gtype)
    f = Constant(mesh, dtype(2.31))
    tol = 50 * np.finfo(gtype).eps
    assert np.isclose(assemble_scalar(f * ufl.dx), f.value, atol=tol)


@pytest.mark.parametrize(
    "gtype, dtype",
    [
        [np.float64, np.float64],
        [np.float64, np.complex128],
        [np.float32, np.float32],
        [np.float32, np.complex64],
    ],
)
def test_assemble_scalar_coefficient(gtype, dtype):
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=gtype)
    V = functionspace(mesh, ("Lagrange", 3))
    u = Function(V, dtype=dtype)
    u.interpolate(lambda x: 3 * x[0] + 2 * x[1] ** 3)
    tol = 50 * np.finfo(gtype).eps
    assert np.isclose(assemble_scalar(u * ufl.dx), gtype(3 / 2 + 2 / 4), atol=tol)


def test_incompatible_coeff_function():
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=np.float64)
    V = functionspace(mesh, ("Lagrange", 3))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: 3 * x[0] + 2 * x[1] ** 3)
    c = Constant(mesh, np.complex128(2.31))
    with pytest.raises(
        RuntimeError, match="All coefficients and constants must have the same data type"
    ):
        assemble_scalar(u * c * ufl.dx)


@pytest.mark.parametrize(
    "gtype, dtype",
    [
        [np.float64, np.float64],
        [np.float64, np.complex128],
        [np.float32, np.float32],
        [np.float32, np.complex64],
    ],
)
@pytest.mark.parametrize("norm_type", ["L2", "H1", "H10", "l2"])
def test_norm(norm_type, dtype, gtype):
    if norm_type == "l2":
        pytest.xfail("Unexpected norm type")

    mesh = create_unit_square(MPI.COMM_WORLD, 3, 5, dtype=gtype)
    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V, dtype=dtype)
    u.interpolate(lambda x: x[0] ** 2 + x[1] ** 2)

    x = ufl.SpatialCoordinate(mesh)
    expr = u - ufl.sin(ufl.pi * x[0])
    compiled_norm = norm(expr, norm_type)

    result = assemble_scalar(compiled_norm)

    if norm_type == "L2":
        ref_form = ufl.inner(expr, expr) * ufl.dx
    elif norm_type == "H1":
        ref_form = (
            ufl.inner(expr, expr) * ufl.dx + ufl.inner(ufl.grad(expr), ufl.grad(expr)) * ufl.dx
        )
    elif norm_type == "H10":
        ref_form = ufl.inner(ufl.grad(expr), ufl.grad(expr)) * ufl.dx

    reference = assemble_scalar(ref_form)
    tol = 50 * np.finfo(dtype).eps
    assert np.isclose(result, reference, atol=tol)


@pytest.mark.skipif(not dolfinx.has_petsc4py, reason="Requires DOLFINX with PETSc4py")
@pytest.mark.skipif(
    hasattr(dolfinx.fem.petsc, "create_matrix_nest"),
    reason="Requires latest version of DOLFINx PETSc API",
)
@pytest.mark.parametrize("kind", [None, "mpi", "nest"])
@pytest.mark.parametrize(
    "alpha", [dolfinx.default_scalar_type(3.0), dolfinx.default_scalar_type(-2.0)]
)
def test_lifting_helper(kind, alpha):
    from petsc4py import PETSc
    import dolfinx.fem.petsc

    mesh = create_unit_square(MPI.COMM_WORLD, 12, 15)
    el0 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    el1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2)

    if kind is None:
        el = basix.ufl.mixed_element([el0, el1])
        W = functionspace(mesh, el)
        V, _ = W.sub(0).collapse()
        Q, _ = W.sub(1).collapse()
    else:
        V = functionspace(mesh, el0)
        Q = functionspace(mesh, el1)
        W = ufl.MixedFunctionSpace(V, Q)

    k = Function(V)
    k.interpolate(lambda x: 3 + x[0] * x[1])
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    if kind is None:
        L = [
            ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), v) * ufl.dx,
            ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)), q) * ufl.dx,
        ]
    else:
        L = [ufl.ZeroBaseForm((v,)), ufl.ZeroBaseForm((q,))]
    a = [
        [k * ufl.inner(u, v) * ufl.dx, None],
        [ufl.inner(u, q) * ufl.dx, k * ufl.inner(p, q) * ufl.dx],
    ]
    if kind is None:
        # Flatten a and L
        L = sum(L)
        a = [sum([aij if aij is not None else 0 for ai in a for aij in ai])]

    L_compiled = form(L)
    a_compiled = form(a)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)

    bc_space = W.sub(1) if kind is None else Q

    bndry_dofs = locate_dofs_topological(bc_space, mesh.topology.dim - 1, bndry_facets)
    bcs = [dirichletbc(default_scalar_type(2.0), bndry_dofs, bc_space)]
    if kind is None:
        lifting_bcs = [bcs]
    else:
        lifting_bcs = bcs

    b = dolfinx.fem.petsc.create_vector(
        dolfinx.fem.forms.extract_function_spaces(L_compiled), kind=kind
    )
    scifem.petsc.zero_petsc_vector(b)
    scifem.petsc.apply_lifting_and_set_bc(b, a_compiled, lifting_bcs, alpha=alpha)

    # Create reference multiplication of alpha Ag
    if kind is None:
        a_compiled = a_compiled[0]
    A = dolfinx.fem.petsc.create_matrix(a_compiled, kind=kind)
    dolfinx.fem.petsc.assemble_matrix(A, a_compiled)
    A.assemble()

    g_ = dolfinx.fem.petsc.create_vector(
        dolfinx.fem.forms.extract_function_spaces(L_compiled), kind=kind
    )
    scifem.petsc.zero_petsc_vector(g_)

    if kind is not None:
        bcs0 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(a_compiled, 0), bcs)
    else:
        bcs0 = bcs
    dolfinx.fem.petsc.set_bc(g_, bcs0)

    b_ref = dolfinx.fem.petsc.create_vector(
        dolfinx.fem.forms.extract_function_spaces(L_compiled), kind=kind
    )
    A.mult(g_, b_ref)
    b_ref.scale(-alpha)
    dolfinx.fem.petsc.set_bc(b_ref, bcs0, alpha=alpha)
    scifem.petsc.ghost_update(b_ref, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
    with b.localForm() as b_local, b_ref.localForm() as b_ref_local:
        np.testing.assert_allclose(b_local.array, b_ref_local.array)
