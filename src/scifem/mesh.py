from __future__ import annotations

from . import _scifem  # type: ignore
import collections
import dolfinx
import typing
import numpy as np
import numpy.typing as npt
from typing import Protocol
from packaging.version import Version
import basix
import ufl

__all__ = [
    "create_entity_markers",
    "transfer_meshtags_to_submesh",
    "SubmeshData",
    "reverse_mark_entities",
    "extract_submesh",
    "find_interface",
    "compute_interface_data",
    "compute_subdomain_exterior_facets",
    "create_geometry_function_space",
    "move",
]

if typing.TYPE_CHECKING:
    TaggedEntities = (
        tuple[int, typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.bool_]]]
        | tuple[int, typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.bool_]], bool]
    )


class _EntityMap(Protocol):
    """Protocol for EntityMap-like objects."""

    sub_topology: dolfinx.mesh.Topology
    dim: int

    def sub_topology_to_topology(
        self, entities: npt.NDArray[np.int32], inverse: bool
    ) -> npt.NDArray[np.int32]:
        """Map entities between sub-topology and topology."""
        ...


def get_entity_map(entity_map: _EntityMap | npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Get an entity map from the sub-topology to the topology.

    This function handles both the deprecated construction of an entity map as a numpy array
    and the newer `EntityMap` class from `dolfinx.mesh`.

    Args:
        entity_map: An `EntityMap` object or a numpy array representing the mapping.
    Returns:
        Mapped indices of entities.
    """
    try:
        sub_top = entity_map.sub_topology
        assert isinstance(sub_top, dolfinx.mesh.Topology)
        sub_map = sub_top.index_map(entity_map.dim)
        indices = np.arange(sub_map.size_local + sub_map.num_ghosts, dtype=np.int32)
        return entity_map.sub_topology_to_topology(indices, inverse=False)
    except AttributeError:
        return entity_map


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

    locate_entities = lambda on_boundary: (
        dolfinx.mesh.locate_entities_boundary if on_boundary else dolfinx.mesh.locate_entities
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
    vertex_to_parent: _EntityMap | npt.NDArray[np.int32],
    cell_to_parent: _EntityMap | npt.NDArray[np.int32],
) -> tuple[dolfinx.mesh.MeshTags, npt.NDArray[np.int32]]:
    """
    Transfer a ``entity_tag`` from a parent mesh to a ``submesh``.

    Args:
        entity_tag: Tag to transfer
        submesh: Submesh to transfer tag to
        vertex_to_parent: Mapping from submesh vertices to parent mesh vertices
        cell_to_parent: Mapping from submesh cells to parent entities
    Returns:
        A tuple (submesh_tag, sub_to_parent_entity_map) where: ``submesh_tag``
        is the tag on the submesh and ``sub_to_parent_entity_map`` is a mapping
        from submesh entities in the tag to the corresponding entities in the parent.
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

    v_to_p = get_entity_map(vertex_to_parent)
    c_to_p = get_entity_map(cell_to_parent)
    cpp_tag, sub_to_parent_entity_map = _scifem.transfer_meshtags_to_submesh_int32(
        entity_tag._cpp_object, submesh.topology._cpp_object, v_to_p, c_to_p
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
    tags_as_arr = np.asarray(tags, dtype=entity_tag.values.dtype)
    all_tagged_indices = np.isin(entity_tag.values, tags_as_arr)
    entities = entity_tag.indices[all_tagged_indices]
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

    reverse_mark_entities(facet_map, interface_facets)

    topology.create_connectivity(tdim - 1, tdim)
    f_to_c = topology.connectivity(tdim - 1, tdim)
    num_cells_per_facet = f_to_c.offsets[interface_facets + 1] - f_to_c.offsets[interface_facets]
    is_interface = interface_facets[num_cells_per_facet == 2]
    return is_interface


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
    submap_array = get_entity_map(cell_map)
    integration_entities[:, 0] = submap_array[integration_entities[:, 0]]

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


def compute_interface_data(
    cell_tags: dolfinx.mesh.MeshTags,
    facet_indices: npt.NDArray[np.int32],
    include_ghosts: bool = False,
) -> npt.NDArray[np.int32]:
    """
    Compute interior facet integrals that are consistently ordered according to the `cell_tags`,
    such that the data `(cell0, facet_idx0, cell1, facet_idx1)` is ordered such that
    `cell_tags[cell0]`<`cell_tags[cell1]`, i.e the cell with the lowest cell marker
    is considered the "+" restriction".

    Args:
        cell_tags: MeshTags that must contain an integer marker for all cells adjacent
            to the `facet_indices`
        facet_indices: List of facets (local index) that are on the interface.
        include_ghosts: If `True` integration entities will include facets that are ghosts on
        the current process. This is for instance useful for interpolation on interior facets.
    Returns:
        The integration data.
    """
    # Future compatibilty check
    integration_args: tuple[int] | tuple
    if Version("0.10.0") <= Version(dolfinx.__version__):
        integration_args = ()
    else:
        fdim = cell_tags.dim - 1
        integration_args = (fdim,)
    if include_ghosts:
        if len(facet_indices) == 0:
            return np.empty((0, 4), dtype=np.int32)
        f_to_c = cell_tags.topology.connectivity(cell_tags.topology.dim - 1, cell_tags.topology.dim)
        c_to_f = cell_tags.topology.connectivity(cell_tags.topology.dim, cell_tags.topology.dim - 1)

        # Extract the cells connected to each facet.
        # Assumption is that there can only be two cells per facet, and should always be
        # two cells per facet.
        num_cells_per_facet = f_to_c.offsets[facet_indices + 1] - f_to_c.offsets[facet_indices]
        assert np.all(num_cells_per_facet == 2), "All facets must be interior facets."
        facet_pos = np.vstack([f_to_c.offsets[facet_indices], f_to_c.offsets[facet_indices] + 1]).T
        cells = f_to_c.array[facet_pos].flatten()
        # Extract facets connected to all cells
        # Assumption is that all cells have the same number of facets
        num_facets_per_cell = c_to_f.offsets[1:] - c_to_f.offsets[:-1]
        assert all(
            num_facets_per_cell[cells.flatten()] == num_facets_per_cell[cells.flatten()[0]]
        ), "Cells must have facets."
        facets = np.vstack(
            [
                c_to_f.array[c_to_f.offsets[cells.flatten()] + i]
                for i in range(num_facets_per_cell[cells.flatten()[0]])
            ]
        ).T
        # Repeat facet indices twice to be able to do vectorized match
        rep_fi = np.repeat(facet_indices, 2)
        indicator = facets == rep_fi[:, None]
        _row, local_pos = np.nonzero(indicator)
        assert np.unique(_row).shape[0] == len(_row)
        idata = np.vstack([cells, local_pos]).T.reshape(-1, 4)
    else:
        idata = dolfinx.cpp.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.interior_facet,
            cell_tags.topology,
            facet_indices,
            *integration_args,
        )
    ordered_idata = idata.reshape(-1, 4).copy()
    switch = cell_tags.values[ordered_idata[:, 0]] > cell_tags.values[ordered_idata[:, 2]]
    if True in switch:
        ordered_idata[switch, :] = ordered_idata[switch][:, [2, 3, 0, 1]]
    return ordered_idata


def create_geometry_function_space(
    mesh: dolfinx.mesh.Mesh, N: int | None = None
) -> dolfinx.fem.FunctionSpace:
    """
    Reconstruct a vector space with N components using
    the geometry dofmap to ensure a 1-1 mapping between mesh nodes and DOFs.

    Args:
        mesh: The mesh to create the function space on.
        N: The number of components. If not provided the geometrical dimension is chosen

    """
    geom_imap = mesh.geometry.index_map()
    geom_dofmap = mesh.geometry.dofmap
    ufl_domain = mesh.ufl_domain()
    assert ufl_domain is not None
    sub_el = ufl_domain.ufl_coordinate_element().sub_elements[0]
    adj_list = dolfinx.cpp.graph.AdjacencyList_int32(geom_dofmap)

    value_shape: tuple[int, ...]
    if N is None:
        ufl_el = basix.ufl.blocked_element(sub_el, shape=(mesh.geometry.dim,))
        value_shape = (mesh.geometry.dim,)
        N = value_shape[0]
    elif N == 1:
        ufl_el = sub_el
        value_shape = ()
    else:
        ufl_el = basix.ufl.blocked_element(sub_el, shape=(N,))
        value_shape = (N,)

    if ufl_el.dtype == np.float32:
        _fe_constructor = dolfinx.cpp.fem.FiniteElement_float32
        _fem_constructor = dolfinx.cpp.fem.FunctionSpace_float32
    elif ufl_el.dtype == np.float64:
        _fe_constructor = dolfinx.cpp.fem.FiniteElement_float64
        _fem_constructor = dolfinx.cpp.fem.FunctionSpace_float64
    else:
        raise RuntimeError(f"Unsupported type {ufl_el.dtype}")
    try:
        cpp_el = _fe_constructor(ufl_el.basix_element._e, block_shape=value_shape, symmetric=False)
    except TypeError:
        cpp_el = _fe_constructor(ufl_el.basix_element._e, block_size=N, symmetric=False)
    dof_layout = dolfinx.cpp.fem.create_element_dof_layout(cpp_el, [])
    cpp_dofmap = dolfinx.cpp.fem.DofMap(dof_layout, geom_imap, N, adj_list, N)

    # Create function space
    try:
        cpp_space = _fem_constructor(mesh._cpp_object, cpp_el, cpp_dofmap)
    except TypeError:
        cpp_space = _fem_constructor(mesh._cpp_object, cpp_el, cpp_dofmap, value_shape=value_shape)

    return dolfinx.fem.FunctionSpace(mesh, ufl_el, cpp_space)


def move(
    mesh: dolfinx.mesh.Mesh,
    u: dolfinx.fem.Function
    | ufl.core.expr.Expr
    | typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.inexact]],
):
    """
    Move the geometry nodes of a mesh given by the movement of a function u.

    Args:
        mesh: The mesh to move
        u: The displacement as a :py:class:dolfinx.fem.Function`,
            :py:class:`ufl.core.expr.Expr` or a lambda function.
    """

    V_geom = create_geometry_function_space(mesh)
    u_geom = dolfinx.fem.Function(V_geom, dtype=mesh.geometry.x.dtype)
    if isinstance(u, dolfinx.fem.Function):
        u_geom.interpolate(u)
    elif isinstance(u, ufl.core.expr.Expr):
        u_compiled = dolfinx.fem.Expression(u, V_geom.element.interpolation_points)
        u_geom.interpolate(u_compiled)
    else:
        u_geom.interpolate(u)
    mesh.geometry.x[:, : mesh.geometry.dim] += u_geom.x.array[:].reshape(-1, mesh.geometry.dim)
