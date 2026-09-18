"""Layer for small backward compatibility wrappers for DOLFINx"""

import numpy.typing as npt
import numpy as np
import dolfinx


def cmap(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.CoordinateElement:
    """Get the basix Cmap for the mesh."""
    if hasattr(mesh.geometry, "cmaps"):
        return mesh.geometry.cmaps[0]
    if callable(mesh.geometry.cmap):
        return mesh.geometry.cmap()
    else:
        return mesh.geometry.cmap


def dofmap(mesh: dolfinx.mesh.Mesh) -> npt.NDArray[np.int32]:
    """Get the dofmap for the geometry."""
    if hasattr(mesh.geometry, "dofmaps"):
        return mesh.geometry.dofmaps[0]
    if callable(mesh.geometry.dofmap):
        return mesh.geometry.dofmap()
    else:
        return mesh.geometry.dofmap


def form_map(form: dolfinx.fem.Form) -> tuple[dolfinx.common.IndexMap, int]:
    try:
        return (
            form.function_spaces[0].dofmaps(0).index_map,
            form.function_spaces[0].dofmaps(0).index_map_bs,
        )

    except TypeError:
        return (
            form.function_spaces[0].dofmaps[0].index_map,
            form.function_spaces[0].dofmaps[0].index_map_bs,
        )


def index_map(comm, size_local, ghosts, owners, tag: int | None = None):
    if dolfinx.common.IndexMap != dolfinx.cpp.common.IndexMap:
        assert tag is not None, "Tag must be provided for dolfinx.common.index_map"
        return dolfinx.common.index_map(comm, size_local, ghosts=(ghosts, owners), tag=tag)
    else:
        try:
            return dolfinx.common.IndexMap(comm, size_local, ghosts, owners)
        except TypeError:
            assert tag is not None, "Tag must be provided for dolfinx.common.IndexMap"
            return dolfinx.common.IndexMap(comm, size_local, ghosts, owners, tag=tag)


def extract_cpp_object(obj):
    if hasattr(obj, "_cpp_object"):
        return obj._cpp_object
    else:
        return obj


def _compat_topology(
    comm, cell_type, tdim, vertex_map, cell_map, c_to_v, v_to_v, original_cell_index
):
    """Construct a ``dolfinx.cpp.mesh.Topology`` across the supported DOLFINx versions.

    The constructor has changed shape more than once and none of the forms is
    introspectable, so they are told apart by the `TypeError` the call itself raises:

    1. communicator and cell type only, everything else through setters;
    2. the same with the maps, dofmap and original cell index passed positionally;
    3. as (2) without the communicator.

    Args:
        comm: The communicator of the new topology.
        cell_type: Its cell type.
        tdim: Its topological dimension, for the index map and connectivity it is set at.
        vertex_map: Index map for dimension 0.
        cell_map: Index map for dimension `tdim`.
        c_to_v: Cell-to-vertex connectivity, in the local numbering of `vertex_map`.
        v_to_v: Vertex-to-vertex connectivity, i.e. the identity over `vertex_map`. Used
            by form (1) only, which cannot derive it.
        original_cell_index: Input global index of each cell. Used by forms (2) and (3),
            which take it directly; form (1) does not accept it.

    Returns:
        The topology, with both index maps and both connectivities set.
    """
    try:
        topology = dolfinx.cpp.mesh.Topology(comm, cell_type)
        topology.set_index_map(0, vertex_map)
        topology.set_index_map(tdim, cell_map)
        topology.set_connectivity(v_to_v, 0, 0)
        topology.set_connectivity(c_to_v, tdim, 0)
        return topology
    except TypeError:
        pass

    args = (
        cell_type,
        extract_cpp_object(vertex_map),
        extract_cpp_object(cell_map),
        extract_cpp_object(c_to_v),
        original_cell_index,
    )
    try:
        return dolfinx.cpp.mesh.Topology(comm, *args)
    except TypeError:
        return dolfinx.cpp.mesh.Topology(*args)
