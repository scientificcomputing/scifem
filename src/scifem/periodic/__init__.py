from scifem.periodic.geometrical_search import match_vertices_geometric
from scifem.periodic.mesh import (
    DEFAULT_TAG_BASE,
    NUM_CONSENSUS_TAGS,
    check_cells_stayed_distinct,
    check_facet_ghosting,
    create_periodic_mesh,
    create_periodic_mesh_from_igi,
)
from scifem.periodic.topological_search import periodic_correspondence_from_nodes
from scifem.periodic.transfer import (
    transfer_function_to_parent_mesh,
    transfer_meshtags_to_periodic_mesh,
)
from scifem.periodic.utils import PeriodicNodes, VertexCorrespondence, resolve_to_roots

__all__ = [
    "DEFAULT_TAG_BASE",
    "NUM_CONSENSUS_TAGS",
    "PeriodicNodes",
    "VertexCorrespondence",
    "check_cells_stayed_distinct",
    "check_facet_ghosting",
    "create_periodic_mesh",
    "create_periodic_mesh_from_igi",
    "match_vertices_geometric",
    "periodic_correspondence_from_nodes",
    "resolve_to_roots",
    "transfer_function_to_parent_mesh",
    "transfer_meshtags_to_periodic_mesh",
]

try:
    from scifem.periodic.gmsh import extract_gmsh_periodic_nodes, read_periodic_mesh_from_msh

    __all__ += ["extract_gmsh_periodic_nodes", "read_periodic_mesh_from_msh"]
except ImportError:
    pass
