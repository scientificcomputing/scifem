import dolfinx
import typing
import numpy as np
import numpy.typing as npt

__all__ = ["create_meshtags"]


# Workaround to make id and locator required but on_boundary optional
# see https://peps.python.org/pep-0655/
class _TaggedEntities(typing.TypedDict):
    tag: int
    locator: typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.bool_]]


class TaggedEntities(_TaggedEntities, total=False):
    on_boundary: bool


def create_meshtags(
    domain: dolfinx.mesh.Mesh,
    dim: int,
    entities_list: list[TaggedEntities],
) -> dolfinx.mesh.MeshTags:
    """Mark entities of specified dimension according to a geometrical marker function.

    Args:
        domain: A ``dolfinx.mesh.Mesh`` object
        dim: Dimension of the entities to mark
        entities_list: A list of dictionaries with the following keys

            - ``tag``: The tag to assign to the entities
            - ``locator``: A function that takes a point and returns a boolean array
              indicating whether the point is inside the entity
            - ``on_boundary``: Optional, if True, the entities will be marked on the boundary

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
        on_boundary = tagged_entity.get("on_boundary", False)
        entities = locate_entities(on_boundary)(domain, dim, tagged_entity["locator"])
        markers[entities] = tagged_entity["tag"]

    facets = np.flatnonzero(markers != -1).astype(np.int32)
    return dolfinx.mesh.meshtags(domain, dim, facets, markers[facets])
