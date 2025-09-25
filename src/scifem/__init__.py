from __future__ import annotations

import dolfinx
import numpy as np
import numpy.typing as npt
from . import _scifem  # type: ignore
from .point_source import PointSource
from .assembly import assemble_scalar, norm
from .bcs import interpolate_function_onto_facet_dofs
from . import xdmf
from .solvers import BlockedNewtonSolver, NewtonSolver
from .spaces import create_real_functionspace, create_space_of_simple_functions
from .mesh import create_entity_markers, transfer_meshtags_to_submesh
from .eval import evaluate_function

__all__ = [
    "PointSource",
    "assemble_scalar",
    "create_space_of_simple_functions",
    "xdmf",
    "create_real_functionspace",
    "assemble_scalar",
    "PointSource",
    "xdmf",
    "vertex_to_dofmap",
    "dof_to_vertexmap",
    "create_entity_markers",
    "NewtonSolver",
    "BlockedNewtonSolver",
    "transfer_meshtags_to_submesh",
    "evaluate_function",
    "norm",
    "interpolate_function_onto_facet_dofs",
]


def vertex_to_dofmap(V: dolfinx.fem.FunctionSpace) -> npt.NDArray[np.int32]:
    """
    Create a map from the vertices (local to the process) to the correspondning degrees
    of freedom.

    Args:
        V: The function space

    Returns:
        An array mapping local vertex i to local degree of freedom

    Note:
        If using a blocked space this map is not unrolled for the DofMap block size.
    """
    return _scifem.vertex_to_dofmap(V.mesh._cpp_object.topology, V.dofmap._cpp_object)


def dof_to_vertexmap(V: dolfinx.fem.FunctionSpace) -> npt.NDArray[np.int32]:
    """
    Create a map from the degrees of freedom to the vertices of the mesh.
    As not every degree of freedom is associated with a vertex, every dof that is not
    associated with a vertex returns -1

    Args:
        V: The function space

    Returns:
        An array mapping local dof i to a local vertex
    """
    num_dofs_local = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    dof_to_vertex_map = np.full(num_dofs_local, -1, dtype=np.int32)
    v_to_d = _scifem.vertex_to_dofmap(V.mesh._cpp_object.topology, V.dofmap._cpp_object)
    dof_to_vertex_map[v_to_d] = np.arange(len(v_to_d), dtype=np.int32)
    return dof_to_vertex_map
