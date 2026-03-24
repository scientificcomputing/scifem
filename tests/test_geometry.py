from mpi4py import MPI
import dolfinx
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scifem import closest_point_projection
from scifem.geometry import project_onto_simplex, _closest_point_projection
import ufl
import basix.ufl
import pytest


def scipy_project_point_to_element(
    mesh: dolfinx.mesh.Mesh,
    cells: npt.NDArray[np.int64],
    target_points: npt.NDArray[np.float64 | np.float32],
    method=None,
    tol: float | None = None,
):
    """
    Projects a 3D point onto a cell in a potentially higher order mesh.

    Args:
        mesh: {py:class}`dolfinx.mesh.Mesh`, the mesh containing the cell.
        cells: {py:class}`numpy.ndarray`, the indices of the cells to project onto.
        target_point: (3,) numpy array, the 3D point to project.
        method: str, the optimization method to use.
        tol: float, the tolerance for the optimizer.

    Returns:
        dict: A dictionary containing the reference coordinates, closest 3D point, and distance.
    """

    # Extract scalar element of mesh
    element = mesh.ufl_domain().ufl_coordinate_element().sub_elements[0]

    # Get the coordinates of the nodes for the specified cell
    node_coords = mesh.geometry.x[mesh.geometry.dofmap[cells]][:, :, : mesh.geometry.dim]
    target_points = target_points.reshape(-1, 3)
    # cmap = mesh.geometry.cmap

    # Constraints and Bounds
    cell_type = mesh.topology.cell_type
    if (
        cell_type == dolfinx.mesh.CellType.triangle
        or cell_type == dolfinx.mesh.CellType.tetrahedron
    ):
        method = method or "SLSQP"
        constraint = {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)}
    else:
        method = method or "L-BFGS-B"
        constraint = {}
    bounds = [(0.0, 1.0) for _ in range(mesh.topology.dim)]

    # Set initial guess and tolerance for solver
    initial_guess = np.full(mesh.topology.dim, 1 / (mesh.topology.dim + 1), dtype=np.float64)
    tol = np.sqrt(np.finfo(mesh.geometry.x.dtype).eps) if tol is None else tol
    closest_points = np.zeros((target_points.shape[0], 3), dtype=mesh.geometry.x.dtype)
    ref_points = np.zeros((target_points.shape[0], mesh.topology.dim), dtype=mesh.geometry.x.dtype)
    for i, (coord, target_point) in enumerate(zip(node_coords, target_points)):
        coord = coord.reshape(-1, mesh.geometry.dim)

        def S(x_ref):
            N_vals = element.tabulate(0, x_ref.reshape(1, mesh.topology.dim))[0, 0, :]
            return np.dot(N_vals, coord)

        def dSdx_ref(x_ref):
            """Evaluate jacobian (tangent vectors) at the given reference coordinates."""
            dN = element.tabulate(1, x_ref.reshape(1, mesh.topology.dim))[
                1 : mesh.topology.dim + 1, 0, :
            ]
            return np.dot(dN, coord)

        def objective(x_ref):
            surface_point = S(x_ref)
            diff = surface_point - target_point
            return 0.5 * np.linalg.norm(diff) ** 2

        def objective_grad(x_ref):
            diff = S(x_ref) - target_point
            tangents = dSdx_ref(x_ref)
            return np.dot(tangents, diff)

        res = minimize(
            objective,
            initial_guess,
            method=method,
            jac=objective_grad,
            bounds=bounds,
            constraints=constraint,
            tol=tol,
            options={"disp": False, "ftol": tol, "maxiter": 250},
        )
        closest_points[i] = S(res.x)
        ref_points[i] = res.x
        assert res.success, f"Optimization failed for {cells[i]} and {target_point=}: {res.message}"
    return closest_points, ref_points


@pytest.mark.parametrize("num_threads", [1, 2, 4])
@pytest.mark.parametrize("order", [1, 2])
def test_2D_manifold(order, num_threads):
    comm = MPI.COMM_SELF

    # Curved quadratic triangle in 3D (6 nodes)
    curved_nodes = np.array(
        [
            [0.0, 0.0, 0.0],  # Node 0: Vertex
            [1.0, 0.0, 0.0],  # Node 1: Vertex
            [0.0, 1.0, 0.0],  # Node 2: Vertex
            [0.6, 0.6, 0.0],  # Node 4: Edge 1-2
            [0.1, 0.5, 0.2],  # Node 5: Edge 2-0 (curved upward in Z)
            [0.5, 0.1, 0.2],  # Node 3: Edge 0-1 (curved upward in Z)
        ]
    )
    cells = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)  # Single curved triangle element
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", order, shape=(3,)))
    if order == 1:
        curved_nodes = curved_nodes[:3]  # Use only vertices for linear case
        cells = cells[:, :3]
    mesh = dolfinx.mesh.create_mesh(comm, cells=cells, x=curved_nodes, e=c_el)

    tol_x = 1e-6
    tol_dist = 1e-7
    theta = np.linspace(0, 4 * np.pi, 1_000_000)
    rand = np.random.RandomState(42)
    R = rand.rand(len(theta))
    z = rand.rand(len(theta)) * 0.5  # Add some random z variation
    points = np.vstack([R * np.cos(theta), R * np.sin(theta), z]).T

    cells = np.zeros(points.shape[0], dtype=np.int32)

    closest_point, closest_ref = closest_point_projection(
        mesh,
        cells,
        points,
        tol_x=tol_x,
        tol_dist=tol_dist,
        tol_grad=1e-16,
        max_iter=2000,
        max_ls_iter=250,
        num_threads=num_threads,
    )

    (result_scipy, ref_scipy) = scipy_project_point_to_element(mesh, cells, points, tol=tol_dist)

    result, ref_coords = _closest_point_projection(
        mesh, cells, points, tol_x=tol_x, tol_dist=tol_dist
    )

    for i, point_to_project in enumerate(points):
        # Check that python and C++ implementations give the same result
        np.testing.assert_allclose(result[i].flatten(), closest_point[i].flatten(), atol=tol_dist)
        np.testing.assert_allclose(
            ref_coords[i].flatten(), closest_ref[i].flatten(), atol=10 * tol_dist
        )

        # Check that we are within the bounds of the simplex
        ref_proj = project_onto_simplex(ref_coords[i])
        np.testing.assert_allclose(ref_proj, ref_coords[i])

        # Check that scipy and our implementation give similar distances,
        # allowing for some tolerance due to different optimization methods
        dist_scipy = 0.5 * np.sum(result_scipy[i] - point_to_project) ** 2
        dist_ours = 0.5 * np.sum(result[i] - point_to_project) ** 2
        if not np.isclose(dist_ours, dist_scipy, atol=tol_dist, rtol=1e-2):
            assert np.linalg.norm(ref_coords[i] - ref_scipy[i]) < 1e-2
        else:
            assert np.isclose(dist_ours, dist_scipy, atol=tol_dist, rtol=1e-2)


@pytest.mark.parametrize("num_threads", [1, 2, 4])
@pytest.mark.parametrize("order", [1, 2])
def test_3D_curved_cell(order, num_threads):
    comm = MPI.COMM_SELF

    curved_nodes_tet = np.array(
        [
            [0.0, 0.0, 0.0],  # 0: Vertex
            [1.0, 0.0, 0.0],  # 1: Vertex
            [0.0, 1.0, 0.0],  # 2: Vertex
            [0.0, 0.0, 1.0],  # 3: Vertex
            [0.0, 0.5, 0.5],  # 4: Edge 2-3
            [0.5, 0.0, 0.5],  # 5: Edge 1-3
            [0.5, 0.5, 0.0],  # 6: Edge 1-2
            [0.0, 0.0, 0.5],  # 9: Edge 0-3
            [0.0, 0.5, 0.2],  # 8: Edge 0-2 (Curved)
            [0.5, 0.0, 0.2],  # 7: Edge 0-1 (Curved)
        ],
        dtype=np.float64,
    )
    cells_tet = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)

    domain_tet = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", order, shape=(3,)))
    if order == 1:
        curved_nodes_tet = curved_nodes_tet[:4]  # Use only vertices for linear case
        cells_tet = cells_tet[:, :4]
    mesh = dolfinx.mesh.create_mesh(comm, cells=cells_tet, x=curved_nodes_tet, e=domain_tet)

    rand = np.random.RandomState(32)
    points = rand.rand(100, 3) - 0.5 * rand.rand(100, 3)
    tol_x = 1e-6
    tol_dist = 1e-7

    cells = np.zeros(points.shape[0], dtype=np.int32)
    closest_point, closest_ref = closest_point_projection(
        mesh,
        cells,
        points,
        tol_x=tol_x,
        tol_dist=tol_dist,
        tol_grad=1e-16,
        max_iter=2000,
        max_ls_iter=250,
        num_threads=num_threads,
    )
    (result_scipy, ref_scipy) = scipy_project_point_to_element(mesh, cells, points, tol=tol_dist)
    result, ref_coords = closest_point_projection(
        mesh,
        cells,
        points,
        tol_x=tol_x,
        tol_dist=tol_dist,
        tol_grad=1e-16,
    )
    for i, point_to_project in enumerate(points):
        np.testing.assert_allclose(result[i].flatten(), closest_point[i].flatten(), atol=tol_dist)
        np.testing.assert_allclose(
            ref_coords[i].flatten(), closest_ref[i].flatten(), atol=10 * tol_x
        )

        # Check that we are within the bounds of the simplex
        ref_proj = project_onto_simplex(ref_coords[i])
        np.testing.assert_allclose(ref_proj, ref_coords[i])

        dist_scipy = 0.5 * np.sum(result_scipy[i] - point_to_project) ** 2
        dist_ours = 0.5 * np.sum(result[i] - point_to_project) ** 2
        if not np.isclose(dist_ours, dist_scipy, atol=tol_dist, rtol=1e-2):
            assert np.linalg.norm(ref_coords[i] - ref_scipy[i]) < 1e-2
        else:
            assert np.isclose(dist_ours, dist_scipy, atol=tol_dist, rtol=1e-2)
