from mpi4py import MPI
import dolfinx
import numpy as np
import numpy.typing as npt
import builtins
import typing
import tempfile

import basix
import ufl
from scipy.optimize import minimize
from ufl.algorithms.signature import compute_expression_signature

T = typing.TypeVar("T", int, float)
MinMaxFunc = typing.Callable[[typing.Sequence[T]], T]


__all__ = ["evaluate_function", "find_cell_extrema", "compute_extrema"]


def _compute_expression_signature(expr: ufl.core.expr.Expr) -> str:
    coeffs = ufl.algorithms.extract_coefficients(expr)
    consts = ufl.algorithms.analysis.extract_constants(expr)
    args = ufl.algorithms.analysis.extract_arguments(expr)
    assert len(args) == 0

    rn = dict()
    rn.update(dict((c, i) for i, c in enumerate(coeffs)))
    rn.update(dict((c, i) for i, c in enumerate(consts)))

    domains: list[ufl.AbstractDomain] = []
    for coeff in coeffs:
        domains.append(*ufl.domain.extract_domains(coeff))
    for gc in ufl.algorithms.analysis.extract_type(expr, ufl.classes.GeometricQuantity):
        domains.append(*ufl.domain.extract_domains(gc))
    for const in consts:
        domains.append(*ufl.domain.extract_domains(const))
    domains = ufl.algorithms.analysis.unique_tuple(domains)
    assert all([isinstance(domain, ufl.Mesh) for domain in domains])
    rn.update(dict((d, i) for i, d in enumerate(domains)))
    return compute_expression_signature(expr, rn)


def evaluate_function(
    u: dolfinx.fem.Function, points: npt.ArrayLike, broadcast=True
) -> npt.NDArray[np.float64]:
    """Evaluate a function at a set of points.

    Args:
        u: The function to evaluate.
        points: The points to evaluate the function at.
        broadcast: If True, the values will be broadcasted to all processes.

            Note:
                Uses a global MPI call to broadcast values, thus this has to
                be called on all active processes synchronously.

            Note:
                If the function is discontinuous, different processes may return
                different values for the same point.
                In this case, the value returned is the maximum value across all processes.

    Returns:
        The values of the function evaluated at the points.


    """
    mesh = u.function_space.mesh
    u.x.scatter_forward()
    comm = mesh.comm
    points = np.array(points, dtype=np.float64)
    assert len(points.shape) == 2, (
        f"Expected points to have shape (num_points, dim), got {points.shape}"
    )
    num_points = points.shape[0]
    extra_dim = 3 - mesh.geometry.dim

    # Append zeros to points if the mesh is not 3D
    if extra_dim > 0:
        points = np.hstack((points, np.zeros((points.shape[0], extra_dim))))

    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    # Find cells whose bounding-box collide with the the points
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    adj = dolfinx.geometry.compute_colliding_cells(mesh, potential_colliding_cells, points)
    indices = np.flatnonzero(adj.offsets[1:] - adj.offsets[:-1])
    cells = adj.array[adj.offsets[indices]]
    points_on_proc = points[indices]

    values = u.eval(points_on_proc, cells)
    if broadcast:
        bs = u.function_space.dofmap.index_map_bs
        # Create array to store values and fill with -inf
        # to ensure that all points are included in the allreduce
        # with op=MPI.MAX
        u_out = np.ones((num_points, bs), dtype=np.float64) * -np.inf
        # Fill in values for points on this process
        u_out[indices, :] = values
        # Now loop over all processes and find the maximum value
        for i in range(num_points):
            if bs > 1:
                # If block size is larger than 1, loop over blocks
                for j in range(bs):
                    u_out[i, j] = comm.allreduce(u_out[i, j], op=MPI.MAX)
            else:
                u_out[i] = comm.allreduce(u_out[i], op=MPI.MAX)

        return u_out
    else:
        return values


def find_cell_extrema(
    u: ufl.core.expr.Expr,
    cell: int,
    kind: MinMaxFunc,
    x0: npt.NDArray | None = None,
    jit_options: dict | None = None,
    method: str | None = None,
    options: dict | None = None,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.floating], np.floating]:
    """
    Find the extrema of a {py:class}`ufl.core.expr.Expr` within a cell.

    Args:
        u: The expression to find the extrema of.
        cell: The local index of the cell to search in
        kind: If we search for minima or maxima
        x_0: The point to start the initial search at
        method: Optimization algorithm to use for local problem
        options: Options for optimization method
        tol: Tolerance for scipy minimize

    Returns:
        The point (in physical space) of the extrema and the value at the extrema.
    """
    if kind is builtins.min:
        sign = 1
    elif kind is builtins.max:
        sign = -1
    else:
        raise NotImplementedError(f"Unknown type of extrema ({kind}), expected {min} or {max}")

    ud = u.ufl_domain()
    assert ud is not None
    mesh = dolfinx.mesh.Mesh(ud.ufl_cargo(), ud)

    if x0 is None:
        x_ref = np.zeros(mesh.topology.dim, dtype=mesh.geometry.x.dtype)
    else:
        x_ref = x0

    _cell = np.array([cell], dtype=np.int32)
    mesh_nodes = mesh.geometry.x[mesh.geometry.dofmap[cell], : mesh.geometry.dim]
    _x_p = np.zeros(3)

    def eval_J(x_ref):
        # Evaluating basis functions through {py:func}`dolfinx.fem.Function.eval`
        # is faster than generating an expression for the same thing
        if isinstance(u, dolfinx.fem.Function):
            # This could in theory be made even faster by taking out some of the eval code
            # However, quite a lot of work needs to be reimplemented for minimal gain
            # to do so, so we rather push forward, then let eval pull back again.
            _x_p[: mesh.geometry.dim] = mesh.geometry.cmap.push_forward(
                x_ref.reshape(-1, mesh.topology.dim), mesh_nodes
            )[0]
            try:
                u_eval = u.eval(_x_p, _cell)[0]
            except RuntimeError:
                # Nonlinear pullback might fail on low precision
                u_expr = dolfinx.fem.Expression(
                    u,
                    x_ref,
                    comm=MPI.COMM_SELF,
                    jit_options=jit_options,
                    dtype=mesh.geometry.x.dtype,
                )
                u_eval = u_expr.eval(mesh, _cell)[0][0]
        else:
            # Expression has to be used for any UFL-expression that is not a Function
            u_expr = dolfinx.fem.Expression(
                u, x_ref, comm=MPI.COMM_SELF, jit_options=jit_options, dtype=mesh.geometry.x.dtype
            )
            u_eval = u_expr.eval(mesh, _cell)[0][0]
        return np.float64(sign * u_eval)  # SLSQP only supports float64

    def eval_dJ(x_ref):
        # Could be improved if we could take `ufl.classes.ReferenceGrad(u)`
        # https://github.com/FEniCS/ufl/issues/450
        u_grad_expr = dolfinx.fem.Expression(
            ufl.Jacobian(mesh).T * ufl.grad(u),
            x_ref,
            comm=MPI.COMM_SELF,
            jit_options=jit_options,
            dtype=mesh.geometry.x.dtype,
        )
        u_grad_eval = u_grad_expr.eval(mesh, _cell)[0][0]
        return (sign * u_grad_eval).astype(np.float64)  # SLSQP only supports float64

    # Bounds force x and y to be between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(mesh.topology.dim)]

    # In SciPy, 'ineq' means the function must be >= 0.
    # So, x + y <= 1 becomes 1 - x - y >= 0.
    cell_type = mesh.topology.cell_type
    if cell_type == dolfinx.mesh.CellType.triangle:
        method = method or "SLSQP"
        constraint = {"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1]}
    elif (
        cell_type == dolfinx.mesh.CellType.quadrilateral
        or cell_type == dolfinx.mesh.CellType.hexahedron
    ):
        method = method or "L-BFGS-B"
        constraint = {}
    elif cell_type == dolfinx.mesh.CellType.tetrahedron:
        method = method or "SLSQP"
        constraint = {"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1] - x[2]}
    else:
        raise RuntimeError(f"Unsupported {cell_type=}")

    # --- 3. Run Optimization ---
    tol = tol if tol is not None else 100 * np.finfo(mesh.geometry.x.dtype).eps
    result = minimize(
        fun=eval_J,
        x0=x_ref.flatten(),
        method=method,
        jac=eval_dJ,
        bounds=bounds,
        constraints=constraint,
        options=options or {},
        tol=tol,
    )

    X_phys = mesh.geometry.cmap.push_forward(result.x.reshape(-1, mesh.topology.dim), mesh_nodes)[0]
    return X_phys, sign * result.fun


def compute_LP_average(
    u: ufl.core.expr.Expr, p: int, domain: dolfinx.mesh.Mesh
) -> npt.NDArray[np.float32 | np.float64 | np.complex128 | np.complex64]:
    """
    Compute the :math:`L^p(\Omega)`-average of a scalar-valued
    {py:class}`ufl-expression<ufl.core.expr.Expr>` over each cell in a
    {py:class}`domain<dolfinx.mesh.Mesh>`

    Args:
        u: The UFL-expression to evaluate
        p: The degree of the norm to use.
        domain: The integration domain

    Returns:
        An array of the local average L^P norm per local cell on the process,
        including ghosts.
    """
    if u.ufl_shape != ():
        raise ValueError(f"u must be scalar-valued, got {u.ufl_shape}.")
    V = dolfinx.fem.functionspace(domain, ("DG", 0))
    v = ufl.TestFunction(V)
    L_p = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(u**p * ufl.conj(v) * ufl.dx, dtype=domain.geometry.x.dtype)
    )
    L_p.scatter_reverse(dolfinx.la.InsertMode.add)
    L_p.scatter_forward()
    cell_vol = dolfinx.fem.assemble_vector(
        dolfinx.fem.form(ufl.conj(v) * ufl.dx, dtype=domain.geometry.x.dtype)
    )
    cell_vol.scatter_reverse(dolfinx.la.InsertMode.add)
    cell_vol.scatter_forward()
    avg = L_p.array / cell_vol.array
    return avg


def compute_extrema(
    u: ufl.core.expr.Expr,
    kind=MinMaxFunc,
    p: int = 1,
    num_candidates: int = 3,
    x0: npt.NDArray[np.floating] | None = None,
    method: str | None = None,
    options: dict | None = None,
    tol: float | None = None,
    jit_options: dict[str, typing.Any] | None = None,
) -> tuple[np.floating, npt.NDArray[np.floating]]:
    """
    Find the extrema of an expression across its integration domain.

    Args:
        u: The expression
        p: Integer to compute the cell-wise L^p norm to speed up search.
        kind: `min` or `max`.
        num_candidates: The number of candidates with the largert average $L^p$
            norm over a cell.
        x0: Initial point in reference cell to start search at.
        method: Optimization algorithm to use for local problem
        options: Options for optimization method
        tol: Tolerance for scipy minimize

    Returns:
        The value at the extrema and the physical point
    """

    # Extract DOLFINx mesh
    ud = u.ufl_domain()
    assert ud is not None
    mesh = dolfinx.mesh.Mesh(ud.ufl_cargo(), ud)

    if kind is builtins.max:
        sign = -1
    elif kind is builtins.min:
        sign = 1
    else:
        raise ValueError("Unknown extrema {kind}, supports min and max only.")

    # Compute cell candidates
    local_avg = compute_LP_average(u, p, mesh)
    order = np.argsort(local_avg)
    if kind is builtins.max:
        candidate_cells = order[-num_candidates:]
    else:
        candidate_cells = order[:num_candidates]

    # Get reference point as midpoint
    if x0 is None:
        vertices = basix.geometry(mesh.basix_cell())
        x0 = np.average(vertices, axis=0)

    # Set max efficiency for temporary expression code.
    jit_options = (
        jit_options.copy()
        if jit_options is not None
        else {
            "cffi_extra_compile_args": ["-march=native", "-O3"],
            "cffi_libraries": ["m"],
        }
    )
    # Set temporary cache dir
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=False, delete=True) as cache_dir:
        jit_options["cache_dir"] = cache_dir
        # Find local extrema
        local_min = np.inf
        local_X = None
        for cell in candidate_cells:
            X_c, extrema = find_cell_extrema(
                u,
                cell,
                kind=kind,
                x0=x0,
                jit_options=jit_options,
                method=method,
                options=options,
                tol=tol,
            )
            if sign * extrema < local_min:
                local_min = sign * extrema
                local_X = X_c

    # Find global extrema
    val, rank = mesh.comm.allreduce((local_min, mesh.comm.rank), op=MPI.MINLOC)
    X = mesh.comm.bcast(local_X, root=rank)
    return sign * val, X
