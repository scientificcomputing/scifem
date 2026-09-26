from mpi4py import MPI
import dolfinx
import pytest
import ufl
import scifem
import numpy as np
import basix
from ffcx.element_interface import map_facet_points

from scifem.bcs import pull_back_to_reference_facet


def right_facets(x):
    return np.isclose(x[0], 1)


@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
        dolfinx.mesh.CellType.hexahedron,
        dolfinx.mesh.CellType.tetrahedron,
    ],
)
@pytest.mark.skipif(
    condition=dolfinx.__version__ == "0.11.0.dev0",
    reason="Update in expression in dolfinx",
)
def test_normal_enforcement(cell_type: dolfinx.mesh.CellType):
    tdim = dolfinx.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 8, 10, 3, cell_type=cell_type)

    V = dolfinx.fem.functionspace(mesh, ("BDM", 1))

    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)

    expr = x[1] * n

    facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, right_facets)
    uh = scifem.interpolate_function_onto_facet_dofs(V, expr, facets)

    tag = 2
    facet_tags = dolfinx.mesh.meshtags(mesh, tdim - 1, facets, np.full_like(facets, tag))
    ds = ufl.ds(domain=mesh, subdomain_data=facet_tags, subdomain_id=tag)
    error = np.sqrt(
        scifem.assemble_scalar(ufl.inner(ufl.dot(uh - expr, n), ufl.dot(uh - expr, n)) * ds)
    )
    assert np.isclose(error, 0.0)


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
@pytest.mark.skipif(
    condition=dolfinx.__version__ == "0.11.0.dev0",
    reason="Update in expression in dolfinx",
)
def test_tangent_enforcement(cell_type: dolfinx.mesh.CellType):
    tdim = dolfinx.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 8, 10, 3, cell_type=cell_type)
    else:
        raise ValueError(f"Unsupported {cell_type=}")

    V = dolfinx.fem.functionspace(mesh, ("N1curl", 2))

    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)

    tangent = ufl.as_vector((-n[1], n[0]))

    expr = (0.1 * x[0] + x[1]) * tangent

    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    uh = scifem.interpolate_function_onto_facet_dofs(V, expr, facets)

    error = np.sqrt(
        scifem.assemble_scalar(
            ufl.inner(ufl.dot(uh - expr, tangent), ufl.dot(uh - expr, tangent)) * ufl.ds
        )
    )
    assert np.isclose(error, 0.0)


@pytest.mark.parametrize(
    "cell_type",
    [
        basix.CellType.triangle,
        basix.CellType.quadrilateral,
        basix.CellType.tetrahedron,
        basix.CellType.hexahedron,
    ],
)
def test_pull_back_to_reference_facet(cell_type: basix.CellType):
    """The pull-back inverts FFCx's map from the reference facet to each facet of the cell."""
    facet_type = basix.cell.subentity_types(cell_type)[-2][0]
    points, _ = basix.make_quadrature(facet_type, 4)
    num_facets = len(basix.topology(cell_type)[-2])
    on_facets = [map_facet_points(points, f, cell_type.name) for f in range(num_facets)]
    pulled_back = pull_back_to_reference_facet(cell_type, on_facets)
    np.testing.assert_allclose(pulled_back, points, atol=1e-14)

    # Points that differ between the facets have no shared facet point set
    on_facets[1] = map_facet_points(points[::-1], 1, cell_type.name)
    with pytest.raises(NotImplementedError, match="differ between the facets"):
        pull_back_to_reference_facet(cell_type, on_facets)
