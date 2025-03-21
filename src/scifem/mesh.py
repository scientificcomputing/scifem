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
