from mpi4py import MPI
import dolfinx
import pytest
import ufl
import scifem
import numpy as np


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
def test_normal_enforcement(cell_type: dolfinx.mesh.CellType):
    tdim = dolfinx.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 8, 10, 3)

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
def test_tangent_enforcement(cell_type: dolfinx.mesh.CellType):
    tdim = dolfinx.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 8, 10, 3)
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
