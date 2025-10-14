import dolfinx
import ufl
import numpy as np
import numpy.typing as npt
from .utils import unroll_dofmap

__all__ = ["interpolation_matrix", "prepare_interpolation_data"]

if dolfinx.has_petsc4py:
    from petsc4py import PETSc

    __all__.append("petsc_interpolation_matrix")


def prepare_interpolation_data(
    expr: ufl.core.expr.Expr, Q: dolfinx.fem.FunctionSpace
) -> npt.NDArray[np.inexact]:
    """Convenience function for preparing data required for assembling the interpolation matrix

    .. math::
        \\begin{align*}
        \\Lambda: V &\\rightarrow Q \\\\
        \\Lambda u &= \\sum_{i=0}^{N_Q-1}\\sum_{j=0}^{N_V-1} \\phi_i l_i(expr(\\psi_j))u_j
        \\end{align*}

    where :math:`l_j` is the dual basis of the space :math:`Q` with basis functions :math:`\\phi_j`,
    and :math:`\\psi_j` are the basis functions of the space :math:`V`.

    Args:
        expr: The UFL expression containing a trial function from space `V`
        Q: Output interpolation space
    Returns:
        Interpolation data per cell, as an numpy array.
    """
    try:
        q_points = Q.element.interpolation_points()
    except TypeError:
        q_points = Q.element.interpolation_points

    arguments = ufl.algorithms.extract_arguments(expr)
    assert len(arguments) == 1
    V = arguments[0].ufl_function_space()

    num_points = q_points.shape[0]
    compiled_expr = dolfinx.fem.Expression(expr, q_points)
    mesh = Q.mesh
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    #
    # (num_cells, num_points, num_dofs*bs, expr_value_size)
    array_evaluated = compiled_expr.eval(mesh, np.arange(num_cells, dtype=np.int32))
    assert np.prod(Q.value_shape) == np.prod(expr.ufl_shape)

    im = Q.element.basix_element.interpolation_matrix

    # Get data as (num_cells*num_points,1, expr_shape, num_test_basis_functions*test_block_size)
    expr_size = int(np.prod(expr.ufl_shape))
    array_evaluated = array_evaluated.reshape(
        num_cells * q_points.shape[0], 1, expr_size, V.dofmap.bs * V.dofmap.dof_layout.num_dofs
    )
    jacobian = dolfinx.fem.Expression(ufl.Jacobian(mesh), q_points)
    detJ = dolfinx.fem.Expression(ufl.JacobianDeterminant(mesh), q_points)
    K = dolfinx.fem.Expression(ufl.JacobianInverse(mesh), q_points)
    jacs = jacobian.eval(mesh, np.arange(num_cells, dtype=np.int32)).reshape(
        num_cells * num_points, mesh.geometry.dim, mesh.topology.dim
    )
    detJs = detJ.eval(mesh, np.arange(num_cells, dtype=np.int32)).flatten()
    Ks = K.eval(mesh, np.arange(num_cells, dtype=np.int32)).reshape(
        num_cells * num_points, mesh.geometry.dim, mesh.topology.dim
    )

    Q_vs = Q.element.basix_element.value_size
    new_array = np.zeros(
        (num_cells * num_points, Q.dofmap.bs * Q_vs, V.dofmap.bs * V.dofmap.dof_layout.num_dofs),
        dtype=np.float64,
    )
    for i in range(V.dofmap.bs * V.dofmap.dof_layout.num_dofs):
        for q in range(Q.dofmap.bs):
            new_array[:, q * Q_vs : (q + 1) * Q_vs, i] = Q.element.basix_element.pull_back(
                array_evaluated[:, :, q * Q_vs : (q + 1) * Q_vs, i], jacs, detJs, Ks
            ).reshape(num_cells * num_points, Q_vs)
    new_array = new_array.reshape(
        num_cells, num_points, Q.dofmap.bs * Q_vs, V.dofmap.bs * V.dofmap.dof_layout.num_dofs
    )
    interpolated_matrix = np.zeros(
        (
            num_cells,
            Q.dofmap.dof_layout.num_dofs * Q.dofmap.bs,
            V.dofmap.bs * V.dofmap.dof_layout.num_dofs,
        ),
        dtype=np.float64,
    )

    for c in range(num_cells):
        for i in range(V.dofmap.bs * V.dofmap.dof_layout.num_dofs):
            tmp_array = np.zeros((int(num_points), Q.dofmap.bs * Q_vs), dtype=np.float64)
            for p in range(num_points):
                tmp_array[p] = new_array[c, p, :, i]
            if Q.dofmap.bs == 1:
                interpolated_matrix[c, :, i] = (im @ tmp_array.T.flatten()).flatten()
            else:
                for q in range(Q.dofmap.bs):
                    interpolated_matrix[c, q :: Q.dofmap.bs, i] = (
                        im @ tmp_array.T[q].flatten()
                    ).flatten()
    # Apply dof transformation to each column (using Piopla maps)
    mesh.topology.create_entity_permutations()
    if Q.element.needs_dof_transformations:
        cell_perm = mesh.topology.get_cell_permutation_info()[:num_cells]

        permuted_matrix = interpolated_matrix.flatten().copy()
        Q.element.Tt_inv_apply(
            permuted_matrix, cell_perm, V.dofmap.bs * V.dofmap.dof_layout.num_dofs
        )
    else:
        permuted_matrix = interpolated_matrix.flatten()
    return permuted_matrix.reshape(interpolated_matrix.shape)


def interpolation_matrix(
    expr: ufl.core.expr.Expr, Q: dolfinx.fem.FunctionSpace
) -> dolfinx.la.MatrixCSR:
    """Create the interpolation matrix :math:`\\Lambda` of a
    :py:class:`UFL-expression<ufl.core.expr.Expr>` such that

    .. math::
        \\begin{align*}
        \\Lambda: V &\\rightarrow Q \\\\
        \\Lambda u &= \\sum_{i=0}^{N_Q-1}\\sum_{j=0}^{N_V-1} \\phi_i l_i(expr(\\psi_j))u_j
        \\end{align*}

    where :math:`l_j` is the dual basis of the space :math:`Q` with
    basis functions :math:`\\phi_j`, and :math:`\\psi_j` are the basis functions of the
    space :math:`V`.

    Args:
        expr: The UFL expression
        Q: Output interpolation space

    Returns:
        Interpolation matrix as a :py:class:`MatrixCSR<dolfinx.la.MatrixCSR>`.
    """

    arguments = ufl.algorithms.extract_arguments(expr)
    assert len(arguments) == 1
    V = arguments[0].ufl_function_space()

    interpolation_data = prepare_interpolation_data(expr, Q)

    q = ufl.TestFunction(Q)
    a = dolfinx.fem.form(ufl.inner(expr, q) * ufl.dx)

    def scatter(
        A: dolfinx.la.MatrixCSR,
        num_cells: int,
        dofs_visited: npt.NDArray[np.int32],
        num_rows_local: int,
        array_evaluated: npt.NDArray[np.inexact],
        dofmap0: npt.NDArray[np.int32],
        dofmap1: npt.NDArray[np.int32],
    ):
        A.data[:] = 0
        for i in range(num_cells):
            rows = dofmap0[i, :]
            cols = dofmap1[i, :]
            A_local = array_evaluated[i].reshape(len(rows), len(cols))
            row_filter = (dofs_visited[rows] == 1) | (rows >= num_rows_local)
            A_local[row_filter] = 0
            A.add(A_local.flatten(), rows, cols)
            dofs_visited[rows] = 1

    A = dolfinx.fem.create_matrix(a)  # , dolfinx.la.BlockMode.expanded)

    row_dofmap = unroll_dofmap(Q.dofmap.list, Q.dofmap.bs)  # (num_cells, num_rows)
    col_dofmap = unroll_dofmap(V.dofmap.list, V.dofmap.bs)  # (num_cells, num_cols)

    num_cells = Q.mesh.topology.index_map(Q.mesh.topology.dim).size_local
    dofs_visited = np.zeros(
        (Q.dofmap.index_map.size_local + Q.dofmap.index_map.num_ghosts) * Q.dofmap.index_map_bs,
        dtype=np.int8,
    )
    num_rows_local = Q.dofmap.index_map.size_local * Q.dofmap.bs
    scatter(A, num_cells, dofs_visited, num_rows_local, interpolation_data, row_dofmap, col_dofmap)
    A.scatter_reverse()
    return A


if dolfinx.has_petsc4py:

    def petsc_interpolation_matrix(
        expr: ufl.core.expr.Expr, Q: dolfinx.fem.FunctionSpace, use_petsc: bool = False
    ) -> PETSc.Mat:
        """Create the interpolation matrix :math:`\\Lambda` of a
        :py:class:`UFL-expression<ufl.core.expr.Expr>` such that

        .. math::
            \\begin{align*}
            \\Lambda: V &\\rightarrow Q \\\\
            \\Lambda u &= \\sum_{i=0}^{N_Q-1}\\sum_{j=0}^{N_V-1} \\phi_i l_i(expr(\\psi_j))u_j
            \\end{align*}

        where :math:`l_j` is the dual basis of the space :math:`Q` with basis
        functions :math:`\\phi_j`, and :math:`\\psi_j` are the basis functions
        of the space :math:`V`.

        Args:
            expr: The UFL expression
            Q: Output interpolation space

        Returns:
            Interpolation matrix as a :py:class:`PETSc.Mat<petsc4py.PETSc.Mat>`.
        """
        arguments = ufl.algorithms.extract_arguments(expr)
        assert len(arguments) == 1
        V = arguments[0].ufl_function_space()

        interpolation_data = prepare_interpolation_data(expr, Q)

        q = ufl.TestFunction(Q)
        a = dolfinx.fem.form(ufl.inner(expr, q) * ufl.dx)
        A = dolfinx.fem.petsc.create_matrix(a)

        def scatter(
            A: PETSc.Mat,
            num_cells: int,
            dofs_visited: npt.NDArray[np.int32],
            num_rows_local: int,
            array_evaluated: npt.NDArray[np.inexact],
            dofmap0: npt.NDArray[np.int32],
            dofmap1: npt.NDArray[np.int32],
        ):
            A.zeroEntries()
            for i in range(num_cells):
                rows = dofmap0[i, :]
                cols = dofmap1[i, :]
                A_local = array_evaluated[i].reshape(len(rows), len(cols))
                row_filter = (dofs_visited[rows] == 1) | (rows >= num_rows_local)
                A_local[row_filter] = 0
                A.setValuesLocal(rows, cols, A_local, addv=PETSc.InsertMode.ADD_VALUES)
                dofs_visited[rows] = 1

        row_dofmap = unroll_dofmap(Q.dofmap.list, Q.dofmap.bs)  # (num_cells, num_rows)
        col_dofmap = unroll_dofmap(V.dofmap.list, V.dofmap.bs)  # (num_cells, num_cols)
        num_cells = Q.mesh.topology.index_map(Q.mesh.topology.dim).size_local
        dofs_visited = np.zeros(
            (Q.dofmap.index_map.size_local + Q.dofmap.index_map.num_ghosts) * Q.dofmap.index_map_bs,
            dtype=np.int8,
        )
        num_rows_local = Q.dofmap.index_map.size_local * Q.dofmap.bs
        scatter(
            A, num_cells, dofs_visited, num_rows_local, interpolation_data, row_dofmap, col_dofmap
        )
        A.assemble()
        return A
