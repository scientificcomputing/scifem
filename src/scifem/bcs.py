import basix
import dolfinx
import ufl
import numpy.typing as npt
import numpy as np

__all__ = ["interpolate_function_onto_facet_dofs"]


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

    # Pull back interpolation points from reference coordinate element to facet reference element
    # We create affine versions of each of them as we are in reference mode.
    # Then we avoid Newton convergence issues
    ref_cmap = basix.ufl.element(
        "Lagrange",
        domain.basix_cell(),
        1,
        shape=(domain.topology.dim,),
        dtype=np.float64,
    )
    ref_top = ref_cmap.reference_topology
    ref_geom = ref_cmap.reference_geometry
    facet_cmap = basix.ufl.element(
        "Lagrange",
        facet_types.pop(),
        1,
        shape=(domain.topology.dim,),
        dtype=np.float64,
    )
    facet_cel = dolfinx.fem.CoordinateElement(
        dolfinx.cpp.fem.CoordinateElement_float64(facet_cmap.basix_element._e)
    )

    reference_facet_points = None
    for i, points in enumerate(interpolation_points[fdim]):
        geom = ref_geom[ref_top[fdim][i]]
        ref_points = facet_cel.pull_back(points, geom)

        # Assert that interpolation points are all equal on all facets
        if reference_facet_points is None:
            reference_facet_points = ref_points
        else:
            assert np.allclose(reference_facet_points, ref_points)
    assert reference_facet_points is not None
    # Create expression for BC
    normal_expr = dolfinx.fem.Expression(expr, reference_facet_points)

    points_per_entity = [sum(ip.shape[0] for ip in ips) for ips in interpolation_points]
    offsets = np.zeros(domain.topology.dim + 2, dtype=np.int32)
    offsets[1:] = np.cumsum(points_per_entity[: domain.topology.dim + 1])
    values_per_entity = np.zeros(
        (offsets[-1], domain.geometry.dim), dtype=dolfinx.default_scalar_type
    )

    # Compute integration entities (cell, local_facet index) for all facets
    all_connected_cells = dolfinx.mesh.compute_incident_entities(
        domain.topology, facets, domain.topology.dim - 1, domain.topology.dim
    )
    values = np.zeros(len(all_connected_cells) * offsets[-1] * domain.geometry.dim)
    domain.topology.create_connectivity(domain.topology.dim, fdim)
    c_to_f = domain.topology.connectivity(domain.topology.dim, fdim)
    num_facets_on_process = (
        domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
    )
    is_marked = np.zeros(num_facets_on_process, dtype=np.int8)
    is_marked[facets] = 1
    for i, cell in enumerate(all_connected_cells):
        values_per_entity[:] = 0.0
        local_facets = c_to_f.links(cell)
        for j, lf in enumerate(local_facets):
            if not is_marked[lf]:
                continue
            insert_pos = offsets[fdim] + reference_facet_points.shape[0] * j
            # Backwards compatibility
            entity = np.array([[cell, j]], dtype=np.int32)
            try:
                normal_on_facet = normal_expr.eval(domain, entity)
            except (AttributeError, AssertionError):
                normal_on_facet = normal_expr.eval(domain, entity.flatten())
            # NOTE: evaluate within loop to avoid large memory requirements
            values_per_entity[insert_pos : insert_pos + reference_facet_points.shape[0]] = (
                normal_on_facet.reshape(-1, domain.geometry.dim)
            )
        values[
            i * offsets[-1] * domain.geometry.dim : (i + 1) * offsets[-1] * domain.geometry.dim
        ] = values_per_entity.reshape(-1)

    qh = dolfinx.fem.Function(Q)
    if hasattr(qh._cpp_object, "interpolate_f"):
        interpolate_func = qh._cpp_object.interpolate_f
    else:
        interpolate_func = qh._cpp_object.interpolate
    interpolate_func(values.reshape(-1, domain.geometry.dim).T.copy(), all_connected_cells)
    qh.x.scatter_forward()

    return qh
