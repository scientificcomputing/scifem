from collections.abc import Sequence

import basix
import dolfinx
import ufl
import numpy.typing as npt
import numpy as np
from packaging.version import Version

from .compat import get_facet_permutations
from .utils import group_by_key
from ffcx.ir.elementtables import (
    permute_quadrature_interval,
    permute_quadrature_triangle,
    permute_quadrature_quadrilateral,
)

__all__ = ["interpolate_function_onto_facet_dofs"]


def build_quadrature_permutations(facet_type, points):
    if Version(dolfinx.__version__) < Version("0.11.0.dev0"):
        # In older versions of dolfinx, the permutation is handled internally
        # in the C++ code, so we can just return the original points
        if facet_type == basix.CellType.interval:
            num_permutations = 2
        elif facet_type == basix.CellType.triangle:
            num_permutations = 6
        elif facet_type == basix.CellType.quadrilateral:
            num_permutations = 8
        else:
            raise ValueError(f"Unsupported {facet_type=}")
        return [points for _ in range(num_permutations)]
    else:
        if facet_type == basix.CellType.interval:
            return [permute_quadrature_interval(points, ref) for ref in range(2)]
        elif facet_type == basix.CellType.triangle:
            perms = []
            # FFCx order: rot is outer loop, ref is inner loop
            for rot in range(3):
                for ref in range(2):
                    # Counteract the mapping with the inverse permutation
                    rot_inv = (3 - rot) % 3 if ref == 0 else rot
                    perms.append(permute_quadrature_triangle(points, ref, rot_inv))
            return perms

        elif facet_type == basix.CellType.quadrilateral:
            perms = []
            # FFCx order: rot is outer loop, ref is inner loop
            for rot in range(4):
                for ref in range(2):
                    # Counteract the mapping with the inverse permutation
                    rot_inv = (4 - rot) % 4 if ref == 0 else rot
                    perms.append(permute_quadrature_quadrilateral(points, ref, rot_inv))
            return perms

        else:
            raise ValueError(f"Unsupported {facet_type=}")


def pull_back_to_reference_facet(
    cell_type: basix.CellType, points_per_facet: Sequence[npt.NDArray[np.floating]]
) -> npt.NDArray[np.floating]:
    """Pull points on the facets of a reference cell back to the reference facet.

    Each facet of a reference cell is an affine image ``x = v_0 + J xi`` of the reference
    facet, so the pull-back ``xi = J^+ (x - v_0)`` is exact.

    Args:
        cell_type: The reference cell.
        points_per_facet: For each local facet, points on it in the cell's reference
            coordinates, shape ``(num_points, tdim)``.

    Returns:
        The points in the reference facet's coordinates, shape ``(num_points, tdim - 1)``,
        which all facets share.

    Raises:
        NotImplementedError: If the facets' points differ in the reference facet's coordinates.
    """
    fdim = len(basix.topology(cell_type)) - 2
    origins = basix.geometry(cell_type)[[f[0] for f in basix.topology(cell_type)[fdim]]]
    jacobians = basix.cell.facet_jacobians(cell_type)
    points = [
        (x - v0) @ np.linalg.pinv(J).T for x, v0, J in zip(points_per_facet, origins, jacobians)
    ]
    if not all(np.allclose(p, points[0]) for p in points):
        raise NotImplementedError(
            "The points differ between the facets of the cell, in reference facet coordinates."
        )
    return points[0]


def interpolate_function_onto_facet_dofs(
    Q: dolfinx.fem.FunctionSpace,
    expr: ufl.core.expr.Expr,
    facets: npt.NDArray[np.int32],
) -> dolfinx.fem.Function:
    """
    Create a function :math:`u_h\\in Q` such that :math:`u_h=\\text{expr}` for all dofs belonging
    to a subset of ``facets``. All other dofs are set to zero.

    Note:
        The resulting function  is only correct in the "normal" direction,
        i.e. :math:`u_{bc}\\cdot n = expr`, while the tangential component is uncontrolled.
        This makes it hard to visualize the function when outputting it to file, either
        through interpolation to an appropriate DG space, or to a point-cloud.

    Args:
        Q: The function space to create the function $u_h$ in.
        expr: The expression to evaluate.
        facets: The facets on which to evaluate the expression.
    """
    domain = Q.mesh
    Q_el = Q.element
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)

    interpolation_points = Q_el.basix_element.x

    facet_types = set(basix.cell.subentity_types(domain.basix_cell())[fdim])
    assert len(facet_types) == 1, "All facets must have the same topology"

    facet_type = facet_types.pop()
    reference_facet_points = pull_back_to_reference_facet(
        domain.basix_cell(), interpolation_points[fdim]
    )
    facet_points = build_quadrature_permutations(facet_type, reference_facet_points)
    expressions = [dolfinx.fem.Expression(expr, points) for points in facet_points]
    points_per_entity = [sum(ip.shape[0] for ip in ips) for ips in interpolation_points]
    offsets = np.zeros(domain.topology.dim + 2, dtype=np.int32)
    offsets[1:] = np.cumsum(points_per_entity[: domain.topology.dim + 1])

    # Compute integration entities (cell, local_facet index) for all facets
    all_connected_cells = dolfinx.mesh.compute_incident_entities(
        domain.topology, facets, domain.topology.dim - 1, domain.topology.dim
    )
    expr_value_size = expressions[0].value_size

    values = np.zeros(len(all_connected_cells) * offsets[-1] * expr_value_size)

    domain.topology.create_connectivity(domain.topology.dim, fdim)
    c_to_f = domain.topology.connectivity(domain.topology.dim, fdim)
    num_facets_on_process = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    is_marked = np.zeros(num_facets_on_process, dtype=np.int8)
    is_marked[facets] = 1
    num_facets_per_cell = dolfinx.cpp.mesh.cell_num_entities(domain.topology.cell_type, fdim)
    facet_permutations = get_facet_permutations(domain.topology)
    # One evaluation per facet permutation, over every marked facet with that permutation
    local_facets = np.asarray(c_to_f.array, dtype=np.int32).reshape(-1, num_facets_per_cell)
    rows, slots = np.nonzero(is_marked[local_facets[all_connected_cells]])
    cells = all_connected_cells[rows]
    permutations = facet_permutations[cells, slots]
    num_points = reference_facet_points.shape[0]
    values = values.reshape(len(all_connected_cells), offsets[-1], expr_value_size)
    for perm, selected in group_by_key(permutations):
        entities = np.column_stack((cells[selected], slots[selected])).astype(np.int32)
        # Backwards compatibility
        try:
            evaluated = expressions[perm].eval(domain, entities)
        except (AttributeError, AssertionError):
            evaluated = expressions[perm].eval(domain, entities.flatten())
        positions = offsets[fdim] + num_points * slots[selected, None] + np.arange(num_points)
        values[rows[selected, None], positions] = evaluated.reshape(
            len(selected), num_points, expr_value_size
        )

    qh = dolfinx.fem.Function(Q)
    if hasattr(qh._cpp_object, "interpolate_f"):
        interpolate_func = qh._cpp_object.interpolate_f
    else:
        interpolate_func = qh._cpp_object.interpolate
    interpolate_func(values.reshape(-1, expr_value_size).T.copy(), all_connected_cells)
    qh.x.scatter_forward()
    return qh
