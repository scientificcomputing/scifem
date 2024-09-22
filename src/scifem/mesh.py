import dolfinx
import typing
import numpy as np
import numpy.typing as npt

__all__ = ["create_meshtags"]


def create_meshtags(
    domain: dolfinx.mesh.Mesh,
    dim: int,
    entities_dict: dict[int, typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.bool_]]],
    on_boundary: bool = False,
) -> dolfinx.mesh.MeshTags:
    """Mark entities of specified dimension according to a geometrical marker function.

    Args:
        domain: A ``dolfinx.mesh.Mesh`` object
        dim: Dimension of the entities to mark
        entities_dict : A dictionary mapping integer tags with a geometrical marker function.
            Example ``{tag: lambda x: x[0]<0}``
        on_boundary: If ``True`` only locate entities on the boundary

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

    if on_boundary:
        locator = dolfinx.mesh.locate_entities_boundary
    else:
        locator = dolfinx.mesh.locate_entities

    # Concatenate and sort the arrays based on indices
    for tag, location in entities_dict.items():
        entities = locator(domain, dim, location)
        markers[entities] = tag

    facets = np.flatnonzero(markers != -1).astype(np.int32)
    return dolfinx.mesh.meshtags(domain, dim, facets, markers[facets])
