# Transfer data between a mesh and its periodic counterpart
# SPDX-License-Identifier: MIT
# Author: Jørgen S. Dokken

"""Transfer data between a mesh and its periodic counterpart"""

import numpy as np
import numpy.typing as npt

import dolfinx

__all__ = [
    "transfer_meshtags_to_periodic_mesh",
    "transfer_function_to_parent_mesh",
]


def transfer_meshtags_to_periodic_mesh(
    mesh: dolfinx.mesh.Mesh,
    periodic_mesh: dolfinx.mesh.Mesh,
    replaced_vertices: npt.NDArray[np.int32],
    meshtags: dolfinx.mesh.MeshTags,
) -> dolfinx.mesh.MeshTags:
    """
    Transfer a mesh tag from a mesh to the periodic mesh.

    Note:
        Entities that have been replaced (vertices, edges, faces) are removed from the mesh tag

    Args:
        mesh: The original mesh
        periodic_mesh: The periodic mesh
        replaced_vertices: The vertices that have been replaced (local to process)
        meshtags: The mesh tag to transfer
    """

    # Remove entities that are fully replaced (all incident vertices replaced).
    if meshtags.dim != mesh.topology.dim:
        mesh.topology.create_connectivity(meshtags.dim, 0)
        e_to_v = mesh.topology.connectivity(meshtags.dim, 0)
        # One cell type, so the offsets are a constant stride and the connectivity can be
        # read as a rectangular array. The assert is where a mixed-topology mesh stops.
        stride = np.diff(e_to_v.offsets)
        assert np.all(stride == stride[:1]), (
            f"entities of dimension {meshtags.dim} do not all have the same number of"
            " vertices, so the connectivity cannot be read as a rectangular array"
        )
        # Dropped when every vertex is replaced: the entity has been merged into its
        # partner, and its input global indices no longer name anything.
        entity_vertices = e_to_v.array.reshape(len(e_to_v.offsets) - 1, -1)
        dropped = np.isin(entity_vertices[meshtags.indices], replaced_vertices).all(axis=1)
        indices = meshtags.indices[~dropped]
        values = meshtags.values[~dropped]
    else:
        indices = meshtags.indices
        values = meshtags.values
    geom_indices = dolfinx.mesh.entities_to_geometry(mesh, meshtags.dim, indices)
    igi_indices = mesh.geometry.input_global_indices[geom_indices]

    periodic_mesh.topology.create_connectivity(mesh.topology.dim, 0)  # This should exist by default
    periodic_mesh.topology.create_entities(meshtags.dim)  # This has to be created
    periodic_mesh.topology.create_connectivity(
        meshtags.dim, 0
    )  # This is requried before distribute entity data
    local_entities, local_values = dolfinx.io.distribute_entity_data(
        periodic_mesh, meshtags.dim, igi_indices, values
    )
    adj = dolfinx.graph.adjacencylist(local_entities)
    return dolfinx.mesh.meshtags_from_entities(
        periodic_mesh, meshtags.dim, adj, local_values.astype(np.int32, copy=False)
    )


def transfer_function_to_parent_mesh(
    u: dolfinx.fem.Function, parent_mesh: dolfinx.mesh.Mesh
) -> dolfinx.fem.Function:
    """
    Transfer a function from a periodic mesh to the ``parent_mesh`` it was created from.

    Use this to visualize a solution. {py:class}`dolfinx.io.VTXWriter` and
    {py:meth}`dolfinx.io.VTKFile.write_function` place one output point per degree of
    freedom, which a periodic mesh cannot supply a coordinate for: a degree of freedom on
    the seam belongs to cells on opposite sides of the domain. On the parent mesh the two
    sides are distinct nodes again, so only the seam is duplicated.

    Args:
        u: The function on the periodic mesh
        parent_mesh: The mesh that was passed to
            {py:func}`scifem.periodic.mesh.create_periodic_mesh`

    Returns:
        A function on ``parent_mesh``, in the same space as ``u``

    Note:
        The transfer is a cell-wise copy, as ``create_periodic_mesh`` leaves the cells and
        the geometry untouched. It therefore requires an element whose dof
        transformations are folded into the dofmap, which is the case for Lagrange and
        discontinuous Lagrange, but not for ``RT``, ``N1curl`` or ``BDM``. Those are not
        accepted by the writers either, so interpolate them first.

    Raises:
        ValueError: If the element of ``u`` needs runtime dof transformations, or if
            ``parent_mesh`` does not have the same cells as the mesh of ``u``
    """
    V = u.function_space
    if V.element.needs_dof_transformations:
        raise ValueError(
            f"Cannot transfer a '{V.ufl_element().family_name}' function cell-wise, as its"
            " dof transformations are not folded into the dofmap. Interpolate into"
            " Lagrange or discontinuous Lagrange first."
        )
    cell_map = parent_mesh.topology.index_map(parent_mesh.topology.dim)
    if cell_map.size_local != V.mesh.topology.index_map(V.mesh.topology.dim).size_local:
        raise ValueError("parent_mesh does not have the same cells as the mesh of u.")

    u_parent = dolfinx.fem.Function(
        dolfinx.fem.functionspace(parent_mesh, V.ufl_element()), name=u.name
    )
    cells = np.arange(cell_map.size_local, dtype=np.int32)
    u_parent.interpolate(u, cells0=cells, cells1=cells)
    u_parent.x.scatter_forward()
    return u_parent
