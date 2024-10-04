from __future__ import annotations

import dolfinx
import typing
import numpy as np
import numpy.typing as npt

__all__ = ["create_entity_markers"]


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
