from __future__ import annotations

from . import _scifem  # type: ignore
import collections
import dolfinx
import typing
import numpy as np
import numpy.typing as npt

__all__ = [
    "create_entity_markers",
    "transfer_meshtags_to_submesh",
    "SubmeshData",
    "reverse_mark_entities",
    "extract_submesh",
    "find_interface",
    "compute_subdomain_exterior_facets",
]


# (tag, locator, on_boundary) where on_boundary is optional
if typing.TYPE_CHECKING:
    TaggedEntities = (
        tuple[int, typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.bool_]]]
        | tuple[int, typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.bool_]], bool]
    )


def create_entity_markers(
    domain: dolfinx.mesh.Mesh,
    dim: int,
    entities_list: list[TaggedEntities],
) -> dolfinx.mesh.MeshTags:
    """Mark entities of specified dimension according to a geometrical marker function.

    Args:
        domain: A ``dolfinx.mesh.Mesh`` object
        dim: Dimension of the entities to mark
        entities_list: A list of tuples with the following elements:

            - ``index 0``: The tag to assign to the entities
            - ``index 1``: A function that takes a point and returns a boolean array
              indicating whether the point is inside the entity
            - ``index 2``: Optional, if True, the entities will be marked on the boundary

    Returns:
        A ``dolfinx.mesh.MeshTags`` object with the corresponding entities marked.
        If an entity satisfies multiple input marker functions,
        it is not deterministic what value the entity gets.

    Note:

    """
    # Create required connectivities
    domain.topology.create_entities(dim)
    domain.topology.create_connectivity(dim, 0)
    domain.topology.create_connectivity(dim, domain.topology.dim)

    # Create marker function
    e_map = domain.topology.index_map(dim)
    num_entities_local = e_map.size_local + e_map.num_ghosts
    markers = np.full(num_entities_local, -1, dtype=np.int32)

    locate_entities = (
        lambda on_boundary: dolfinx.mesh.locate_entities_boundary
        if on_boundary
        else dolfinx.mesh.locate_entities
    )

    # Concatenate and sort the arrays based on indices
    for tagged_entity in entities_list:
        on_boundary = False if len(tagged_entity) == 2 else tagged_entity[2]
        entities = locate_entities(on_boundary)(domain, dim, tagged_entity[1])
        markers[entities] = tagged_entity[0]

    facets = np.flatnonzero(markers != -1).astype(np.int32)
    return dolfinx.mesh.meshtags(domain, dim, facets, markers[facets])


def transfer_meshtags_to_submesh(
    entity_tag: dolfinx.mesh.MeshTags,
    submesh: dolfinx.mesh.Mesh,
    vertex_to_parent: npt.NDArray[np.int32],
    cell_to_parent: npt.NDArray[np.int32],
) -> tuple[dolfinx.mesh.MeshTags, npt.NDArray[np.int32]]:
    """
    Transfer a ``entity_tag`` from a parent mesh to a ``submesh``.

    Args:
        entity_tag: Tag to transfer
        submesh: Submesh to transfer tag to
        vertex_to_parent: Mapping from submesh vertices to parent mesh vertices
        cell_to_parent: Mapping from submesh cells to parent entities
    Returns:
        A tuple (submesh_tag, sub_to_parent_entity_map) where: ``submesh_tag`` is the tag on the
        submesh and ``sub_to_parent_entity_map`` is a mapping from submesh entities in the tag to
        the corresponding entities in the parent.
    """
    dim = entity_tag.dim
    sub_tdim = submesh.topology.dim
    if dim > sub_tdim:
        raise RuntimeError(
            f"Cannot transfer meshtags of dimension {dim} to submesh with topological dimension"
        )

    submesh.topology.create_connectivity(sub_tdim, sub_tdim)
    submesh.topology.create_connectivity(entity_tag.dim, 0)
    submesh.topology.create_connectivity(sub_tdim, entity_tag.dim)
    entity_tag.topology.create_connectivity(dim, 0)
    entity_tag.topology.create_connectivity(dim, sub_tdim)
    cpp_tag, sub_to_parent_entity_map = _scifem.transfer_meshtags_to_submesh_int32(
        entity_tag._cpp_object, submesh.topology._cpp_object, vertex_to_parent, cell_to_parent
    )
    return dolfinx.mesh.MeshTags(cpp_tag), sub_to_parent_entity_map


def reverse_mark_entities(
    entity_map: dolfinx.common.IndexMap, entities: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
    """Communicate entities marked on a single process to all processes that ghosts or
    owns this entity.

    Args:
        entity_map: Index-map describing entity ownership
        entities: Local indices of entities to communicate
    Returns:
        Local indices marked on any process sharing this entity
    """
    comm_vec = dolfinx.la.vector(entity_map, dtype=np.int32)
    comm_vec.array[:] = 0
    comm_vec.array[entities] = 1
    comm_vec.scatter_reverse(dolfinx.la.InsertMode.add)
    comm_vec.scatter_forward()
    return np.flatnonzero(comm_vec.array).astype(np.int32)


SubmeshData = collections.namedtuple(
    "SubmeshData", ["domain", "cell_map", "vertex_map", "node_map", "cell_tag"]
)


def extract_submesh(
    mesh: dolfinx.mesh.Mesh, entity_tag: dolfinx.mesh.MeshTags, tags: typing.Sequence[int]
) -> SubmeshData:
    """Generate a sub-mesh from a subset of tagged entities in a meshtag object.

    Args:
        mesh: The mesh to extract the submesh from.
        entity_tag: MeshTags object containing marked entities.
        tags: What tags the marked entities used in the submesh should have.

    Returns:
        A tuple `(submesh, subcell_to_parent_entity, subvertex_to_parent_vertex,
        subnode_to_parent_node, entity_tag_on_submesh)`.
    """

    # Accumulate all entities, including ghosts, for the specfic set of tagged entities
    edim = entity_tag.dim
    mesh.topology.create_connectivity(edim, mesh.topology.dim)
    emap = mesh.topology.index_map(entity_tag.dim)
    marker = dolfinx.la.vector(emap)
    tags_as_arr = np.asarray(tags, dtype=entity_tag.values.dtype)
    all_tagged_indices = np.isin(entity_tag.values, tags_as_arr)
    marker.array[entity_tag.indices[all_tagged_indices]] = 1
    marker.scatter_reverse(dolfinx.la.InsertMode.add)
    marker.scatter_forward()
    entities = np.flatnonzero(marker.array)
    # Extract submesh
    submesh, cell_map, vertex_map, node_map = dolfinx.mesh.create_submesh(mesh, edim, entities)

    # Transfer cell markers
    new_et, _ = transfer_meshtags_to_submesh(entity_tag, submesh, vertex_map, cell_map)
    new_et.name = entity_tag.name
    return SubmeshData(submesh, cell_map, vertex_map, node_map, new_et)


def find_interface(
    cell_tags: dolfinx.mesh.MeshTags,
    id_0: tuple[int, ...],
    id_1: tuple[int, ...],
) -> npt.NDArray[np.int32]:
    """Given to sets of cells, find the facets that are shared between them.

    Args:
        cell_tags: MeshTags object marking cells.
        id_0: Tags to extract for domain 0
        id_1: Tags to extract for domain 1

    Returns:
        The facets shared between the two domains.
    """
    topology = dolfinx.mesh.Topology(cell_tags.topology)

    assert topology.dim == cell_tags.dim
    tdim = topology.dim
    cell_map = topology.index_map(tdim)

    # Find all cells on process that has cell with tag(s) id_0
    domain_0 = reverse_mark_entities(
        cell_map,
        cell_tags.indices[
            np.isin(cell_tags.values, np.asarray(id_0, dtype=cell_tags.values.dtype))
        ],
    )

    # Find all cells on process that has cell with tag(s) id_1
    domain_1 = reverse_mark_entities(
        cell_map,
        cell_tags.indices[
            np.isin(cell_tags.values, np.asarray(id_1, dtype=cell_tags.values.dtype))
        ],
    )

    # Find all facets connected to each domain
    topology.create_connectivity(tdim, tdim - 1)
    facet_map = topology.index_map(tdim - 1)

    local_facets0 = dolfinx.mesh.compute_incident_entities(topology, domain_0, tdim, tdim - 1)
    facets0 = reverse_mark_entities(facet_map, local_facets0)

    local_facets1 = dolfinx.mesh.compute_incident_entities(topology, domain_1, tdim, tdim - 1)
    facets1 = reverse_mark_entities(facet_map, local_facets1)

    # Compute intersecting facets
    interface_facets = np.intersect1d(facets0, facets1)
    return reverse_mark_entities(facet_map, interface_facets)


def compute_subdomain_exterior_facets(
    mesh: dolfinx.mesh.Mesh, ct: dolfinx.mesh.MeshTags, markers: typing.Sequence[int]
) -> npt.NDArray[np.int32]:
    """Find the the facets that are considered to be on the "exterior" boundary of a subdomain.

    The subdomain is defined as the collection of cells in ``ct`` that is marked with any of the
    ``markers``. The exterior boundary of the subdomain is defined as the collection of facets
    that are only connected to a single cell within the subdomain.

    Note:
        Ghosted facets are included in the resulting array.

    Args:
        mesh: Mesh to extract subdomains from
        ct: MeshTags object marking subdomains
        markers: The tags making up the "new" mesh
    Returns:
        The exterior facets
    """
    # Create submesh to find the exterior facet of subdomain
    sub_mesh, cell_map, _, _, _ = extract_submesh(
        mesh,
        ct,
        markers,
    )
    sub_mesh.topology.create_connectivity(sub_mesh.topology.dim - 1, sub_mesh.topology.dim)
    sub_facets = dolfinx.mesh.exterior_facet_indices(sub_mesh.topology)

    # Map exterior facet to (submesh_cell, local_facet_index) tuples
    try:
        integration_entities = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.exterior_facet, sub_mesh.topology, sub_facets
        )
    except TypeError:
        integration_entities = dolfinx.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.exterior_facet,
            sub_mesh.topology,
            sub_facets,
            sub_mesh.topology.dim - 1,
        )
    integration_entities = integration_entities.reshape(-1, 2)
    # Map submesh_cell to parent cell
    integration_entities[:, 0] = cell_map[integration_entities[:, 0]]

    # Get cell to facet connectivity (parent mesh)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    num_facets_per_cell = dolfinx.cpp.mesh.cell_num_entities(
        mesh.topology.cell_type, mesh.topology.dim - 1
    )
    c_to_f = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 1).array.reshape(
        -1, num_facets_per_cell
    )
    # Map (parent_cell, local_facet_index) to facet index (local to process)
    parent_facets = c_to_f[integration_entities[:, 0], integration_entities[:, 1]]
    facet_map = mesh.topology.index_map(mesh.topology.dim - 1)
    # Accumulate ghost facets
    return reverse_mark_entities(facet_map, parent_facets)
