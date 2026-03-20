import dolfinx
import numpy as np
import numpy.typing as npt
import warnings


def project_onto_simplex(
    v: npt.NDArray[np.float64 | np.float32],
) -> npt.NDArray[np.float64 | np.float32]:
    """
    Exact projection of vector v onto the simplex {x >= 0, sum(x) <= 1}.

    See Algorithm 1: Laurent Condat. Fast Projection onto the Simplex and the l1 Ball.
    Mathematical Programming, Series A, 2016, 158 (1), pp.575-585.
    ⟨DOI: 10.1007/s10107-015-0946-6⟩.


    Args:
        v: The vector to project onto simplex

    Return:
        The projection of v onto the simplex.

    """
    # 1. First try the unconstrained positive quadrant projection
    v = v.reshape(-1)
    u = np.maximum(v, 0.0)
    if np.sum(u) <= 1.0:
        return u

    # 2. Otherwise, project exactly onto the slanted face sum(x) = 1
    tdim = len(v)
    if tdim == 2:
        sort_v = np.array([max(u[0], u[1]), min(u[0], u[1])])
        cssv = np.array([sort_v[0], sort_v[0] + sort_v[1]])
    elif tdim == 3:
        sort_v = np.sort(v)[::-1]
        cssv = np.array([sort_v[0], sort_v[0] + sort_v[1], sort_v[0] + sort_v[1] + sort_v[2]])
    else:
        raise RuntimeError("Projection onto simplex is only implemented for 2D and 3D vectors.")
    # cssv = np.cumsum(sort_v)

    # Find the primal-dual root
    # sum x_i = a
    # Find K: = max (k in [1,.. N] such that sum_{r=1}^k sort_v[r] - a)/k < u_k
    # Multiply by k and take the last true entry to get the index rho
    K = np.nonzero(sort_v * np.arange(1, tdim + 1) > (cssv - 1.0))[0][-1]
    tau = (cssv[K] - 1.0) / (K + 1.0)

    return np.maximum(v - tau, 0.0)


def closest_point_projection(
    mesh: dolfinx.mesh.Mesh,
    cells: npt.NDArray[np.int32],
    target_points: npt.NDArray[np.float64 | np.float32],
    tol_x: float | None = None,
    tol_dist: float = 1e-10,
    tol_grad: float = 1e-10,
    max_iter: int = 2000,
    max_ls_iter: int = 250,
) -> tuple[npt.NDArray[np.float64 | np.float32], npt.NDArray[np.float64 | np.float32]]:
    """
    Projects a 3D point onto a cell in a potentially higher order mesh.

    Uses the Goldstein-Levitin-Polyak Gradient projection method, where
    potential simplex constraints are handled by an exact projection using a
    primal-dual root finding method. See:
    - Held, M., Wolfe, P., Crowder, H.: Validation of subgradient optimization (1974)
    - Laurent Condat. Fast Projection onto the Simplex and the l1 Ball. (2016)
    - Dimitri P. Bertsekas, "On the Goldstein-Levitin-Polyak gradient projection method," (1976)

    Args:
        mesh: {py:class}`dolfinx.mesh.Mesh`, the mesh containing the cell.
        cells: {py:class}`numpy.ndarray`, the local indices of the cells to project onto.
        target_point: (3,) numpy array, the 3D point to project.
        tol_x: Tolerance for changes between iterates in the reference coordinates.
        If None, uses the square root of machine precision.
        tol_dist: Tolerance used to determine if the projected point is close enough to
            the target point to stop optimization.
        max_iter: int, the maximum number of iterations for the projected gradient method.
        max_ls_iter: int, the maximum number of line search iterations.

    Returns:
        A tuple of arrays containing the closest points (in physical space)
        and reference coordinates for each cell to each target point.
    """
    dtype = mesh.geometry.x.dtype
    eps = np.finfo(dtype).eps
    tol_x = np.sqrt(eps) if tol_x is None else tol_x
    roundoff_tol = 100 * eps

    # Extract scalar element of mesh
    element = mesh.ufl_domain().ufl_coordinate_element().sub_elements[0]
    tdim = mesh.topology.dim
    # Get the coordinates of the nodes for the specified cell
    node_coords = mesh.geometry.x[mesh.geometry.dofmap[cells]][:, :, : mesh.geometry.dim]
    target_points = target_points.reshape(-1, 3)
    # cmap = mesh.geometry.cmap

    # Constraints and Bounds
    cell_type = mesh.topology.cell_type

    # Set initial guess and tolerance for solver
    initial_guess = np.full(mesh.topology.dim, 1 / (mesh.topology.dim + 1), dtype=dtype)
    closest_points = np.zeros((target_points.shape[0], 3), dtype=dtype)
    reference_points = np.zeros((target_points.shape[0], mesh.topology.dim), dtype=dtype)
    is_simplex = cell_type in [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.tetrahedron,
    ]

    if is_simplex:

        def project(x):
            return project_onto_simplex(x)
    else:

        def project(x):
            return np.clip(x, 0.0, 1.0)

    for i, (coord, target_point) in enumerate(zip(node_coords, target_points)):
        coord = coord.reshape(-1, mesh.geometry.dim)
        x_k = initial_guess.copy()

        for k in range(max_iter):
            x_old = x_k.copy()

            # 1. Evaluate ONLY the Value and First Derivative (Gradient)
            tab = element.tabulate(1, x_k.reshape(1, tdim))
            surface_point = np.dot(tab[0, 0, :], coord)
            diff = surface_point - target_point
            current_dist_sq = 0.5 * np.linalg.norm(diff) ** 2

            tangents = np.dot(tab[1 : tdim + 1, 0, :], coord)
            g = np.dot(tangents, diff)

            # SCALED GRADIENT TOLERANCE
            jac_norm = np.linalg.norm(tangents)
            scaled_tol_grad = tol_grad * max(jac_norm, 1.0)
            if np.linalg.norm(g) < scaled_tol_grad:
                break
            # 2. First-Order Step Direction
            p = -g

            # 3. Goldstein-Polyak-Levitin Projected Line Search
            # Bertsekas (1976) Eq. (14) - Armijo Rule along the Projection Arc
            sigma = 0.1  # Sufficient decrease parameter (0 < sigma < 0.5)
            beta = 0.5  # Reduction factor (0 < beta < 1)
            alpha = 1.0  # Initial step size

            x_new_prev = np.full(tdim, -1, dtype=dtype)
            target_reached = False
            for li in range(max_ls_iter):
                # Apply the exact analytical simplex projection
                x_new = project(x_k + alpha * p)

                if np.linalg.norm(x_new - x_new_prev) < eps:
                    # The projection is pinned to a boundary.
                    # Changing alpha further will not change the physical point!
                    break
                x_new_prev = x_new.copy()
                # The actual physical step we took after hitting the geometric walls
                actual_step = x_new - x_k

                # Evaluate distance at the projected point
                tab_new = element.tabulate(0, x_new.reshape(1, tdim))
                S_new = np.dot(tab_new[0, 0, :], coord)
                new_dist_sq = 0.5 * np.linalg.norm(S_new - target_point) ** 2
                if new_dist_sq < 0.5 * tol_dist**2:
                    # We are close enough to the target point, no need for further line search
                    target_reached = True
                    break

                # Bertsekas Eq. (14) condition:
                # f(x_new) <= f(x_k) + sigma * grad_f(x_k)^T * (x_new - x_k)
                # Note: g is grad_f(x_k)
                if new_dist_sq <= current_dist_sq + sigma * np.dot(g, actual_step) + roundoff_tol:
                    # Condition satisfied
                    break

                # Reduction step (Backtracking)
                alpha *= beta

            if li == max_ls_iter - 1:
                warnings.warn(
                    f"Line search failed to converge after {max_ls_iter} iterations "
                    + f"for cell {cells[i]}  and {target_point=}."
                )
            x_k[:] = x_new

            if target_reached:
                break

            # 4. Check for convergence
            if np.linalg.norm(x_k - x_old) < tol_x:
                break
            if new_dist_sq < 0.5 * tol_dist**2:
                print("Projected point is within tolerance of target point, stopping optimization.")
                break

        assert np.allclose(project(x_k), x_k), "Projection failed to satisfy constraints"

        # Final coordinate extraction
        tab_final = element.tabulate(0, x_k.reshape(1, tdim))
        closest_points[i] = np.dot(tab_final[0, 0, :], coord)
        reference_points[i] = x_k

        if k == max_iter - 1:
            raise RuntimeError(
                f"Projected gradient method failed to converge after {max_iter} iterations ",
                f"for cell {cells[i]} and {target_point=} and final iterate {x_k=} ",
                f"and final point {closest_points[i]} with final distance ",
                f"{np.linalg.norm(closest_points[i] - target_point)}.",
            )
    return closest_points, reference_points
