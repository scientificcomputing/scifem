import typing

from mpi4py import MPI

from packaging.version import Version
import basix
import dolfinx
import ufl
import numpy as np
import numpy.typing as npt
import scipy.linalg
from .bcs import build_quadrature_permutations, pull_back_to_reference_facet
from .compat import create_cell_permutations, get_facet_permutations
from .ufl_compat import apply_pullback_inverse
from .utils import group_by_key, unroll_dofmap
from .mesh import _EntityMap

__all__ = [
    "interpolation_matrix",
    "prepare_interpolation_data",
    "interpolate_to_surface_submesh",
    "interpolate_from_surface_submesh",
    "SurfaceSubmeshInterpolation",
    "SurfaceSubmeshExtension",
    "compute_entity_closure_permutations",
    "compute_entity_closure_dofs",
]

if dolfinx.has_petsc4py:
    from petsc4py import PETSc

    __all__.append("petsc_interpolation_matrix")


def prepare_interpolation_data(
    expr: ufl.core.expr.Expr,
    Q: dolfinx.fem.FunctionSpace,
    interpolation_entities: npt.NDArray[np.int32] | None = None,
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
        interpolation_entities: Entities of the domain of the input space `V` that one
            should evaluate the `expr` at. If not provided, it is assumed that
            we are integrating over all cells in `V` and that `Q` is defined on the same grid.
    Returns:
        Interpolation data per cell, as an numpy array.
    """
    if np.issubdtype(dolfinx.default_scalar_type, np.complexfloating):
        raise NotImplementedError("No complex support")

    # Extract argument from expr (in V)
    arguments = ufl.algorithms.extract_arguments(expr)
    assert len(arguments) == 1
    V = arguments[0].ufl_function_space()

    mesh = V.mesh

    if Q.mesh.topology.dim == V.mesh.topology.dim:
        if interpolation_entities is None:
            tdim = mesh.topology.dim
            num_cells = mesh.topology.index_map(tdim).size_local
            interpolation_entities = np.arange(num_cells, dtype=np.int32)
        else:
            if (ndim := interpolation_entities.ndim) != 1:
                raise ValueError(
                    f"Interpolation entities has wrong input shape, should be 1D, got {ndim}"
                )
            num_cells = len(interpolation_entities)
    elif Q.mesh.topology.dim == V.mesh.topology.dim - 1:
        if interpolation_entities is None:
            raise ValueError(
                "For integration onto a submesh of codim 1,"
                + "the integration entities has to be provided"
            )
        else:
            if (ndim := interpolation_entities.ndim) != 2:
                raise ValueError(
                    f"Interpolation entities has wrong input shape, should be 2D, got {ndim}"
                )
            num_cells = interpolation_entities.shape[0]
    else:
        raise RuntimeError("Only codim-1 interpolation matrices can be defined")

    # Extract quadrature points for expression (in Q space)
    try:
        q_points = Q.element.interpolation_points()
    except TypeError:
        q_points = Q.element.interpolation_points

    # Compile expression
    num_points = q_points.shape[0]
    compiled_expr = dolfinx.fem.Expression(expr, q_points)

    # (num_cells, num_points, num_dofs*bs, expr_value_size)
    array_evaluated = compiled_expr.eval(mesh, interpolation_entities)
    assert np.prod(Q.value_shape) == np.prod(expr.ufl_shape)

    # Get data as (num_cells*num_points,1, expr_shape, num_test_basis_functions*test_block_size)
    expr_size = int(np.prod(expr.ufl_shape))
    array_evaluated = array_evaluated.reshape(
        num_cells * q_points.shape[0], 1, expr_size, V.dofmap.bs * V.dofmap.dof_layout.num_dofs
    )

    # Check if we are dealing with a quadrature element or not.
    # They do not have a complete DOLFINx API, which makes them tricky to use.
    try:
        basix_el = Q.element.basix_element
        Q_vs = basix_el.value_size
        pull_back = basix_el.pull_back
        im = basix_el.interpolation_matrix
    except (RuntimeError, ValueError):
        Q_vs = 1  # If we do not have a basix element, assume value size is 1
        assert isinstance(Q.ufl_element().pullback, ufl.pullback.IdentityPullback)
        pull_back = lambda x: None
        assert Q.element.interpolation_ident
        im = None

    new_array = np.zeros(
        (num_cells * num_points, Q.dofmap.bs * Q_vs, V.dofmap.bs * V.dofmap.dof_layout.num_dofs),
        dtype=np.float64,
    )

    # Check if pullback is identity, then we can skip this step
    if not isinstance(Q.ufl_element().pullback, ufl.pullback.IdentityPullback):
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

        for i in range(V.dofmap.bs * V.dofmap.dof_layout.num_dofs):
            for q in range(Q.dofmap.bs):
                new_array[:, q * Q_vs : (q + 1) * Q_vs, i] = pull_back(
                    array_evaluated[:, :, q * Q_vs : (q + 1) * Q_vs, i], jacs, detJs, Ks
                ).reshape(num_cells * num_points, Q_vs)
        new_array = new_array.reshape(
            num_cells, num_points, Q.dofmap.bs * Q_vs, V.dofmap.bs * V.dofmap.dof_layout.num_dofs
        )
    else:
        new_array = array_evaluated.reshape(
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
    # Check if interpolation matrix of dual operator is identity, then we can use a vectorized
    # version of this step
    if Q.element.interpolation_ident:
        # Smart vectorized version with identity mapping
        if Q.dofmap.bs == 1:
            interpolated_matrix = new_array.transpose(0, 2, 1, 3).reshape(
                new_array.shape[0], new_array.shape[1] * new_array.shape[2], new_array.shape[3]
            )
        else:
            i_scalar = new_array.transpose(0, 2, 1, 3)
            interpolated_matrix = np.zeros(
                (new_array.shape[0], new_array.shape[1] * new_array.shape[2], new_array.shape[3])
            )
            for q in range(Q.dofmap.bs):
                interpolated_matrix[:, q :: Q.dofmap.bs, :] = i_scalar[:, q, :, :]

    else:
        # Tedious non-identity version
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

    if Q.element.needs_dof_transformations:
        # Apply dof transformation to each column (using Piola maps)
        create_cell_permutations(mesh.topology)
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


def compute_entity_closure_permutations(
    V: dolfinx.fem.FunctionSpace,
    dim: int,
    entities: npt.NDArray[np.int32],
    size: int | None = None,
) -> npt.NDArray[np.int32]:
    """Permutations taking each entity's closure dofs from its cell's orientation to its own.

    Row ``i`` reorders data given at the closure dofs of the local entity ``entities[i, 1]`` of
    cell ``entities[i, 0]``, in the order of that entity in the cell, into the order of the entity
    as a cell of its own (as in a submesh), with its vertices ordered by global index:
    ``data_entity = data_cell[perm[i]]``. Every cell sharing an entity therefore agrees on the
    order. It is basix's ``permute_subentity_closure_inv`` with the cell's permutation info,
    computed once per distinct ``(cell permutation info, local entity)`` pair.

    Args:
        V: Space on the parent mesh whose element defines the permutations.
        dim: Topological dimension of the entities, below that of the cells.
        entities: ``(cell, local entity)`` pairs, shape ``(num_entities, 2)``, such as facet
            integration entities for ``dim = tdim - 1``.
        size: Length of each permutation. Defaults to the number of closure dofs of an entity.

    Returns:
        The permutations, shape ``(num_entities, size)``.

    Raises:
        ValueError: If ``size`` is not given and the entities' closures differ in size, as for the
            triangular and quadrilateral facets of a prism.
    """
    mesh = V.mesh
    element = V.element.basix_element
    entity_types = basix.cell.subentity_types(element.cell_type)[dim]
    create_cell_permutations(mesh.topology)
    cell_info = mesh.topology.get_cell_permutation_info()
    entities = np.asarray(entities, dtype=np.int32).reshape(-1, 2)
    cells, local_entities = entities.T
    if size is None:
        layout = V.dofmap.dof_layout
        sizes = {len(layout.entity_closure_dofs(dim, int(e))) for e in np.unique(local_entities)}
        if len(sizes) > 1:
            raise ValueError(f"The entities' closures have different numbers of dofs: {sizes}.")
        size = sizes.pop() if sizes else 0

    permutations = np.empty((len(entities), size), dtype=np.int32)
    keys = cell_info[cells].astype(np.int64) * (int(local_entities.max(initial=0)) + 1)
    keys += local_entities
    for _, rows in group_by_key(keys):
        entity = int(local_entities[rows[0]])
        permutation = np.arange(size, dtype=np.int32)
        element.permute_subentity_closure_inv(
            permutation, int(cell_info[cells[rows[0]]]), entity_types[entity], entity
        )
        permutations[rows] = permutation
    return permutations


def compute_entity_closure_dofs(
    V: dolfinx.fem.FunctionSpace, dim: int, entities: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """The dofs of each entity's closure, from its cell's dofmap, in the entity's own orientation.

    Entry ``[i, j]`` is the dof of ``V`` (a node, for a blocked space) at the ``j``-th closure dof
    of the local entity ``entities[i, 1]`` of cell ``entities[i, 0]``, taken as a cell of its own.
    Cells sharing an entity give the same row, and for facets it lines up with the dofs of the same
    element's trace on a facet submesh. See :py:func:`compute_entity_closure_permutations`.

    Args:
        V: Space on the parent mesh.
        dim: Topological dimension of the entities, below that of the cells.
        entities: ``(cell, local entity)`` pairs, shape ``(num_entities, 2)``.

    Returns:
        The dofs, shape ``(num_entities, num closure dofs)``.
    """
    entities = np.asarray(entities, dtype=np.int32).reshape(-1, 2)
    layout = V.dofmap.dof_layout
    local = np.take_along_axis(
        np.array([layout.entity_closure_dofs(dim, int(e)) for e in entities[:, 1]], dtype=np.int32)
        if len(entities)
        else np.zeros((0, 0), dtype=np.int32),
        compute_entity_closure_permutations(V, dim, entities),
        axis=1,
    )
    return np.take_along_axis(V.dofmap.list[entities[:, 0]], local, axis=1)


class SurfaceSubmeshInterpolation:
    """Interpolation of a volume expression into a space on a facet submesh, prepared once.

    The expression is compiled into a :py:class:`dolfinx.fem.Expression` at the surface element's
    interpolation points, and the connectivities and permutation info it needs are created, once.
    :py:meth:`apply` then evaluates the expression on the parent facets, at the current values of
    its coefficients, and interpolates the result into each submesh cell. For DOLFINx < 0.11
    the values also have to be reordered from each facet's orientation in its cell to its own
    (:py:func:`compute_entity_closure_permutations`); that permutation is computed here too.

    Note:
        Does not work for DG as no dofs are associated with the facets in versions of DOLFINx
        prior to https://github.com/FEniCS/dolfinx/pull/4140, which is included in version
        0.11.0 and later.

    Args:
        expr: Function or expression on the parent mesh to interpolate from
        V_surface: Space on the facet submesh to interpolate into
        submesh_facets: Cells in facet mesh
        integration_entities: Integration entities on the parent mesh
            corresponding to the facets in `submesh_facets`
        entity_maps: Entity maps for an expression with coefficients on other meshes
    """

    def __init__(
        self,
        expr: ufl.core.expr.Expr,
        V_surface: dolfinx.fem.FunctionSpace,
        submesh_facets: npt.NDArray[np.int32],
        integration_entities: npt.NDArray[np.int32],
        entity_maps: list[_EntityMap] | None = None,
    ):
        if Version(dolfinx.__version__) < Version("0.10.0"):
            raise RuntimeError("interpolate_to_submesh requires dolfinx version 0.10.0 or higher")
        ufl_domains = ufl.domain.extract_domains(expr)
        max_tdim_pos = np.argmax([domain.ufl_cargo().topology.dim for domain in ufl_domains])
        self._mesh = dolfinx.mesh.Mesh(
            ufl_domains[max_tdim_pos].ufl_cargo(), ufl_domains[max_tdim_pos]
        )
        self._V_surface = V_surface
        self._submesh_facets = submesh_facets
        self._integration_entities = integration_entities

        submesh = V_surface.mesh
        ip = V_surface.element.interpolation_points
        try:
            self._expression = dolfinx.fem.Expression(expr, ip, entity_maps=entity_maps)
        except TypeError:
            self._expression = dolfinx.fem.Expression(expr, ip)
        self._mesh.topology.create_connectivity(self._mesh.topology.dim, submesh.topology.dim)
        self._mesh.topology.create_connectivity(submesh.topology.dim, self._mesh.topology.dim)
        create_cell_permutations(self._mesh.topology)
        create_cell_permutations(submesh.topology)
        # Before the introduction of https://github.com/FEniCS/dolfinx/pull/4140
        # one needed to permute the data according to the facet permutations,
        # one permutation per evaluation point.
        self._permutations: npt.NDArray[np.int32] | None = None
        if Version(dolfinx.__version__) < Version("0.11.0.dev0"):
            V_vol = expr.function_space  # type: ignore[attr-defined]
            self._permutations = compute_entity_closure_permutations(
                V_vol, V_vol.mesh.topology.dim - 1, integration_entities, ip.shape[0]
            )

    @property
    def V_surface(self) -> dolfinx.fem.FunctionSpace:
        """The space on the facet submesh interpolated into."""
        return self._V_surface

    @property
    def expression(self) -> dolfinx.fem.Expression:
        """The compiled expression, at the surface element's interpolation points."""
        return self._expression

    def apply(self, u_surface: dolfinx.fem.Function):
        """Interpolate the expression, at its coefficients' current values, into ``u_surface``.

        Args:
            u_surface: Function in :py:attr:`V_surface`, overwritten on the submesh cells, and
                scattered forward.
        """
        data = self._expression.eval(self._mesh, self._integration_entities)
        if self._permutations is not None:
            data = data[np.arange(data.shape[0])[:, None], self._permutations]

        if len(data.shape) == 3:
            # Data is now (num_cells, value_size,num_points)
            data = data.swapaxes(1, 2)
            # Data is now (value_size, num_cells, num_points)
            data = data.swapaxes(0, 1)

        if self._expression.value_size == 1:
            shaped_data = data.flatten()
        else:
            shaped_data = data.reshape(self._expression.value_size, -1)

        if hasattr(u_surface._cpp_object, "interpolate_f"):
            interpolate_func = u_surface._cpp_object.interpolate_f
        else:
            interpolate_func = u_surface._cpp_object.interpolate

        interpolate_func(shaped_data, self._submesh_facets)
        u_surface.x.scatter_forward()


def interpolate_to_surface_submesh(
    u_volume: dolfinx.fem.Function,
    u_surface: dolfinx.fem.Function,
    submesh_facets: npt.NDArray[np.int32],
    integration_entities: npt.NDArray[np.int32],
    entity_maps: list[_EntityMap] | None = None,
):
    """
    Interpolate a function `u_volume` into the function `u_surface`.

    See :py:class:`SurfaceSubmeshInterpolation`, which prepares the interpolation once for
    repeated use.

    Note:
        Does not work for DG as no dofs are associated with the facets in versions of DOLFINx
        prior to https://github.com/FEniCS/dolfinx/pull/4140, which is included in version
        0.11.0 and later.

    Args:
        u_volume: Function to interpolate data from
        u_surface: Function to interpolate data to
        submesh_facets: Cells in facet mesh
        integration_entities: Integration entities on the parent mesh
            corresponding to the facets in `submesh_facets`
        entity_maps: Entity maps for an expression with coefficients on other meshes
    """
    SurfaceSubmeshInterpolation(
        u_volume, u_surface.function_space, submesh_facets, integration_entities, entity_maps
    ).apply(u_surface)


class SurfaceSubmeshExtension:
    """Extension by zero of a function on a facet submesh into a space on its parent mesh.

    The reverse of :py:class:`SurfaceSubmeshInterpolation`: the dofs on the closure of each
    submesh facet are set from the surface function and all other volume dofs are zero. Dofs
    shared by several facets get the mean of the facets' values (see :py:meth:`apply`).

    Supported volume spaces are continuous Lagrange and, given ``entity_maps``, Piola-mapped
    spaces such as RT and N1curl, of which only the facet trace (the normal or tangential
    component) is set.

    For a surface space that is the trace of the volume space, this is a right inverse of
    :py:func:`interpolate_to_surface_submesh`.

    Args:
        V_surface: A space on the facet submesh: with as many components as ``V_volume`` for
            Lagrange, and vector-valued of the geometric dimension otherwise.
        V_volume: A continuous space on the parent mesh.
        submesh_facets: Cells of the submesh to extend from.
        integration_entities: ``(cell, local facet)`` on the parent mesh of each of
            ``submesh_facets``, as for :py:func:`interpolate_to_surface_submesh`.
        entity_maps: The submesh's entity map, needed for a volume space that is not Lagrange.

    Raises:
        ValueError: If ``V_volume`` is discontinuous, the value sizes do not fit, or
            ``entity_maps`` is missing where it is needed.
        NotImplementedError: If ``V_surface`` needs dof transformations, the volume element's
            pullback cannot be inverted, or its interpolation points differ between the facets
            of a cell.
    """

    _V_surface: dolfinx.fem.FunctionSpace
    _V_volume: dolfinx.fem.FunctionSpace
    _basis: npt.NDArray[np.floating]
    _surface_dofs: npt.NDArray[np.int32]
    _volume_dofs: npt.NDArray[np.int32]
    _weights: npt.NDArray[np.floating]

    def __init__(
        self,
        V_surface: dolfinx.fem.FunctionSpace,
        V_volume: dolfinx.fem.FunctionSpace,
        submesh_facets: npt.NDArray[np.int32],
        integration_entities: npt.NDArray[np.int32],
        entity_maps: list[_EntityMap] | None = None,
    ):
        self._V_surface, self._V_volume = V_surface, V_volume
        element = V_volume.element.basix_element
        if element.discontinuous:
            raise ValueError("The volume space must be continuous.")
        if V_surface.element.needs_dof_transformations:
            raise NotImplementedError(
                "Surface spaces that need dof transformations are not supported."
            )
        submesh_facets = np.asarray(submesh_facets, dtype=np.int32)
        if element.family == basix.ElementFamily.P and element.interpolation_is_identity:
            self._basis, self._volume_dofs = self._setup_nodal(submesh_facets, integration_entities)
        else:
            self._basis, self._volume_dofs = self._setup_piola(integration_entities, entity_maps)
        s_bs = V_surface.dofmap.bs
        cell_dofs = V_surface.dofmap.list[submesh_facets]
        self._surface_dofs = (cell_dofs[:, :, None] * s_bs + np.arange(s_bs)).reshape(
            len(submesh_facets), -1
        )
        self._weights = self._inverse_write_counts(self._volume_dofs)

    def _setup_nodal(
        self, submesh_facets: npt.NDArray[np.int32], integration_entities: npt.NDArray[np.int32]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.int32]]:
        """The surface basis at the Lagrange nodes of each facet's closure, and those nodes'
        volume dofs, unrolled by block."""
        V_surface, V_volume = self._V_surface, self._V_volume
        element = V_volume.element.basix_element
        bs = V_volume.dofmap.index_map_bs
        value_size = int(np.prod(V_surface.value_shape, dtype=int))
        if value_size != bs:
            raise ValueError(
                f"The surface space has {value_size} components, the volume space {bs}."
            )

        # The volume element's points on the reference facet, in the facet's closure order, and
        # the volume dofs at them, in the orientation of each submesh cell
        facet_type = V_surface.element.basix_element.cell_type
        trace = basix.create_element(
            element.family, facet_type, element.degree, element.lagrange_variant
        )
        points = trace.points
        nodes = compute_entity_closure_dofs(
            V_volume, V_volume.mesh.topology.dim - 1, integration_entities
        )
        assert nodes.shape[1] == len(points), "The facet trace does not match the closure dofs"
        volume_dofs = (nodes[:, :, None] * bs + np.arange(bs)).reshape(len(nodes), -1)

        # Surface basis at those points, (facets, points * bs, surface dofs per cell)
        u = ufl.TestFunction(V_surface)
        basis = dolfinx.fem.Expression(u, points).eval(V_surface.mesh, submesh_facets)
        return basis.reshape(len(submesh_facets), len(points) * bs, -1), volume_dofs

    def _inverse_write_counts(self, volume_dofs: npt.NDArray[np.int32]) -> npt.NDArray[np.floating]:
        """One over the number of writes to each of ``volume_dofs``, over all facets and
        processes, with the same shape."""
        count = dolfinx.fem.Function(self._V_volume)
        count.x.array[:] = 0.0
        np.add.at(count.x.array, volume_dofs.reshape(-1), 1.0)
        count.x.scatter_reverse(dolfinx.la.InsertMode.add)
        count.x.scatter_forward()
        return 1.0 / count.x.array[volume_dofs]

    def _setup_piola(
        self,
        integration_entities: npt.NDArray[np.int32],
        entity_maps: list[_EntityMap] | None,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.int32]]:
        """The map from the surface dofs to the volume dofs of each facet's closure, for a
        Piola-mapped volume space, and those volume dofs.

        The pulled-back surface basis is evaluated at the volume element's interpolation points
        on the closure, on the parent ``(cell, local facet)`` entities through ``entity_maps``.
        The interpolation matrix, restricted to the closure (it is block-diagonal by entity),
        turns it into the cell's reference dofs, and the dof transformations into global ones.
        """
        V_surface, V_volume = self._V_surface, self._V_volume
        mesh = V_volume.mesh
        element = V_volume.element.basix_element
        if entity_maps is None:
            raise ValueError(
                "A Piola-mapped volume space needs entity_maps, to evaluate the surface function "
                "on the parent mesh's facets."
            )
        if tuple(V_surface.value_shape) != (mesh.geometry.dim,):
            raise ValueError(
                f"The surface space must be vector-valued, of shape ({mesh.geometry.dim},), to be "
                f"pulled back; it has shape {tuple(V_surface.value_shape)}."
            )
        fdim = mesh.topology.dim - 1
        facet_types = set(basix.cell.subentity_types(element.cell_type)[fdim])
        if len(facet_types) != 1:
            raise NotImplementedError("Cells with facets of different types are not supported.")
        facet_type = facet_types.pop()

        # The interpolation points and dofs of each local facet's closure, in the cell's order
        connectivity = basix.cell.sub_entity_connectivity(element.cell_type)
        # Per sub-entity: interpolation points, dofs and interpolation matrix, which basix stores
        # as (dofs, value components, points, derivatives)
        entity_points = typing.cast(list[list[npt.NDArray[np.floating]]], element.x)
        entity_matrices = typing.cast(list[list[npt.NDArray[np.floating]]], element.M)
        if any(m.shape[-1] != 1 for matrices in entity_matrices for m in matrices):
            raise NotImplementedError("Elements whose dofs involve derivatives are not supported.")
        closure_dofs, closure_points, matrices = [], [], []
        num_facets_per_cell = dolfinx.cpp.mesh.cell_num_entities(mesh.topology.cell_type, fdim)
        for facet in range(num_facets_per_cell):
            closure = [(d, e) for d in range(fdim + 1) for e in connectivity[fdim][facet][d]]
            closure_dofs.append(
                np.concatenate([element.entity_dofs[d][e] for d, e in closure]).astype(np.int32)
            )
            # Each entity's dofs depend only on its own points, so the closure's matrix is the
            # entities' blocks on its diagonal, as (dofs, points * components)
            blocks = [entity_matrices[d][e][..., 0].transpose(0, 2, 1) for d, e in closure]
            blocks = [b.reshape(b.shape[0], b.shape[1] * b.shape[2]) for b in blocks]
            matrices.append(scipy.linalg.block_diag(*blocks))
            closure_points.append(np.vstack([entity_points[d][e] for d, e in closure]))
        # One Expression per facet permutation needs the same facet points for every facet
        reference_points = pull_back_to_reference_facet(element.cell_type, closure_points)

        entities = np.asarray(integration_entities, dtype=np.int32).reshape(-1, 2)
        cells, local_facets = entities.T
        create_cell_permutations(mesh.topology)
        cell_info = mesh.topology.get_cell_permutation_info()[cells]
        permutations = get_facet_permutations(mesh.topology)[cells, local_facets]
        groups = group_by_key(permutations.astype(np.int64) * num_facets_per_cell + local_facets)

        v = ufl.TestFunction(V_surface)
        expr = apply_pullback_inverse(V_volume.ufl_element().pullback, v, mesh.ufl_domain())
        point_sets = build_quadrature_permutations(facet_type, reference_points)
        # Compiled on each process for its own permutations only
        expressions = {
            perm: dolfinx.fem.Expression(
                expr,
                point_sets[perm],
                comm=MPI.COMM_SELF,
                entity_maps=entity_maps,  # type: ignore[arg-type]
            )
            for perm in {int(key) // num_facets_per_cell for key, _ in groups}
        }

        num_surface_dofs = V_surface.dofmap.dof_layout.num_dofs * V_surface.dofmap.bs
        cell_dofs = V_volume.dofmap.list[cells]
        shape = (len(entities), len(closure_dofs[0]))
        basis = np.empty((*shape, num_surface_dofs), dtype=dolfinx.default_scalar_type)
        volume_dofs = np.empty(shape, dtype=np.int32)
        for key, rows in groups:
            perm, facet = divmod(int(key), num_facets_per_cell)
            values = expressions[perm].eval(mesh, entities[rows])
            values = values.reshape(len(rows), -1, num_surface_dofs)
            # To the global orientation, on full cell arrays: the transformations are
            # block-diagonal by entity, so the other dofs stay zero.
            cell_values = np.zeros((len(rows), element.dim, num_surface_dofs), dtype=basis.dtype)
            cell_values[:, closure_dofs[facet]] = np.einsum("cq,rqd->rcd", matrices[facet], values)
            flat = cell_values.reshape(-1)
            V_volume.element.Tt_inv_apply(flat, cell_info[rows], num_surface_dofs)
            basis[rows] = flat.reshape(cell_values.shape)[:, closure_dofs[facet]]
            volume_dofs[rows] = cell_dofs[rows][:, closure_dofs[facet]]
        return basis, volume_dofs

    @property
    def V_surface(self) -> dolfinx.fem.FunctionSpace:
        """The surface space."""
        return self._V_surface

    @property
    def V_volume(self) -> dolfinx.fem.FunctionSpace:
        """The volume space."""
        return self._V_volume

    @property
    def basis(self) -> npt.NDArray[np.floating]:
        """The map from each facet's :py:attr:`surface_dofs` to its :py:attr:`volume_dofs`,
        ``(facets, volume dofs, surface dofs)``."""
        return self._basis

    @property
    def surface_dofs(self) -> npt.NDArray[np.int32]:
        """The surface dofs of each facet's submesh cell, unrolled by block, ``(facets, surface
        dofs)``."""
        return self._surface_dofs

    @property
    def volume_dofs(self) -> npt.NDArray[np.int32]:
        """The volume dofs of each facet's closure, local to the process and unrolled by block,
        ``(facets, volume dofs)``."""
        return self._volume_dofs

    @property
    def weights(self) -> npt.NDArray[np.floating]:
        """One over the number of writes to each of :py:attr:`volume_dofs`, over all facets and
        processes, with the same shape."""
        return self._weights

    def apply(self, u_surface: dolfinx.fem.Function, u_volume: dolfinx.fem.Function):
        """Set ``u_volume`` to the extension of ``u_surface``.

        Unlike :py:func:`interpolate_to_surface_submesh` and
        :py:func:`scifem.interpolate_function_onto_facet_dofs`, this does not call
        :py:meth:`dolfinx.fem.Function.interpolate`, which would set every dof of each parent cell
        and zero boundary nodes that lie on none of that cell's facets. The values are instead
        added straight into the volume dofs of each facet's closure, scaled by :py:attr:`weights`,
        then summed onto their owners with a reverse scatter, i.e.
        ``u_volume[volume_dofs] += weights * (basis @ u_surface[surface_dofs])``. Each boundary
        dof thus gets the
        mean of its writes: its value for a compatible (continuous) surface space, and an average
        of the facets' values where they disagree.

        Args:
            u_surface: Function in :py:attr:`V_surface`, with consistent ghosts.
            u_volume: Function in :py:attr:`V_volume`, overwritten, ghosts included.
        """
        values = np.einsum("fnd,fd->fn", self._basis, u_surface.x.array[self._surface_dofs])
        u_volume.x.array[:] = 0.0
        np.add.at(u_volume.x.array, self._volume_dofs.reshape(-1), (values * self._weights).ravel())
        u_volume.x.scatter_reverse(dolfinx.la.InsertMode.add)
        u_volume.x.scatter_forward()


def interpolate_from_surface_submesh(
    u_surface: dolfinx.fem.Function,
    u_volume: dolfinx.fem.Function,
    submesh_facets: npt.NDArray[np.int32],
    integration_entities: npt.NDArray[np.int32],
    entity_maps: list[_EntityMap] | None = None,
):
    """
    Extend a function ``u_surface`` on a facet submesh by zero into the function ``u_volume``.

    The reverse of :py:func:`interpolate_to_surface_submesh`: ``u_volume`` takes the values of
    ``u_surface`` at its nodes on the facets and is zero at every other node. See
    :py:class:`SurfaceSubmeshExtension`, which can be reused across calls.

    Args:
        u_surface: Function on the facet submesh to extend
        u_volume: Function on the parent mesh to extend into, continuous Lagrange or Piola-mapped
        submesh_facets: Cells in facet mesh
        integration_entities: Integration entities on the parent mesh
            corresponding to the facets in `submesh_facets`
        entity_maps: The submesh's entity map, needed for a Piola-mapped ``u_volume``
    """
    SurfaceSubmeshExtension(
        u_surface.function_space,
        u_volume.function_space,
        submesh_facets,
        integration_entities,
        entity_maps,
    ).apply(u_surface, u_volume)
