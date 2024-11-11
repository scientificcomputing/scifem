# Create a periodic mesh in parallel
# SPDX-License-Identifier: MIT
# Author: Jørgen S. Dokken

from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import numpy.typing as npt


mpi_dtype = {np.float64: MPI.DOUBLE, np.float32: MPI.FLOAT,
             np.int32: MPI.INT32_T, np.int64: MPI.INT64_T,
             np.complex128: MPI.DOUBLE_COMPLEX, np.complex64: MPI.COMPLEX}


def transfer_meshtags_to_periodic_mesh(mesh: dolfinx.mesh.Mesh, periodic_mesh:dolfinx.mesh.Mesh,
                                       replaced_vertices: npt.NDArray[np.int32],
                                        meshtags:dolfinx.mesh.MeshTags)->dolfinx.mesh.MeshTags:
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

    # Remove entities that have been replaced (vertices, edges, faces)
    if meshtags.dim != mesh.topology.dim:
        mesh.topology.create_connectivity(meshtags.dim, 0)
        e_to_v = mesh.topology.connectivity(meshtags.dim, 0)
        e_to_v_new = e_to_v.array.copy()
        replacement_indicator = np.isin(e_to_v_new, replaced_vertices)
        e_to_v_new[replacement_indicator] = -1
        new_adj = dolfinx.graph.adjacencylist(e_to_v_new, e_to_v.offsets)
        indices = []
        values = []
        for entity, value in zip(meshtags.indices, meshtags.values):
            if not np.allclose(new_adj.links(entity), -1):
                indices.append(entity)
                values.append(value)
        indices = np.array(indices, dtype=np.int32)
        values = np.array(values, dtype=meshtags.values.dtype)
    else:
        indices = meshtags.indices
        values = tags_old.values
    geom_indices = dolfinx.mesh.entities_to_geometry(mesh, meshtags.dim, indices)
    igi_indices = mesh.geometry.input_global_indices[geom_indices]

    local_entities, local_values = dolfinx.io.distribute_entity_data(
        periodic_mesh, meshtags.dim, igi_indices, values
    )
    periodic_mesh.topology.create_connectivity(mesh.topology.dim, 0)
    adj = dolfinx.graph.adjacencylist(local_entities)
    periodic_mesh.topology.create_entities(meshtags.dim)
    return dolfinx.mesh.meshtags_from_entities(
            periodic_mesh, meshtags.dim, adj, local_values.astype(np.int32, copy=False)
        )


def all_to_allv(comm, send_data, num_send_data, recv_data, num_recv_data):
    dtype = mpi_dtype[send_data.dtype.type]
    assert recv_data.dtype == send_data.dtype, f"Data types do not match, {recv_data.dtype} != {send_data.dtype}"
    assert (d_size:=send_data.size) ==  (s_size:=num_send_data.sum()), f"Number of send data {d_size}  does not match data size {s_size}"
    assert (d_size:=recv_data.size) ==  (r_size:=num_recv_data.sum()), f"Number of recv data {d_size}  does not match data size {r_size}"

    send_msg = [send_data, num_send_data, dtype]
    recv_msg = [recv_data, num_recv_data, dtype]
    comm.Neighbor_alltoallv(send_msg, recv_msg)


def get_ownership(imap)->npt.NDArray[np.int32]:
    """
    Get ownership of each index in an index map
    """
    owners = np.full(imap.size_local + imap.num_ghosts, imap.comm.rank, dtype=np.int32)
    owners[imap.size_local:] = imap.owners
    return owners

def compute_insert_position(
    data_owner: npt.NDArray[np.int32],
    destination_ranks: npt.NDArray[np.int32],
    out_size: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """
    Giving a list of ranks, compute the local insert position for each rank in a list
    sorted by destination ranks. This function is used for packing data from a
    given process to its destination processes.

    Example:

        .. highlight:: python
        .. code-block:: python

            data_owner = [0, 1, 1, 0, 2, 3]
            destination_ranks = [2,0,3,1]
            out_size = [1, 2, 1, 2]
            insert_position = compute_insert_position(data_owner, destination_ranks, out_size)

        Insert position is then ``[1, 4, 5, 2, 0, 3]``
    """
    process_pos_indicator = data_owner.reshape(-1, 1) == destination_ranks

    # Compute offsets for insertion based on input size
    send_offsets = np.zeros(len(out_size) + 1, dtype=np.intc)
    send_offsets[1:] = np.cumsum(out_size)
    assert send_offsets[-1] == len(data_owner)

    # Compute local insert index on each process
    proc_row, proc_col = np.nonzero(process_pos_indicator)
    cum_pos = np.cumsum(process_pos_indicator, axis=0)
    insert_position = cum_pos[proc_row, proc_col] - 1

    # Add process offset for each local index
    insert_position += send_offsets[proc_col]
    return insert_position

def unroll_insert_position(
    insert_position: npt.NDArray[np.int32], block_size: int
) -> npt.NDArray[np.int32]:
    """
    Unroll insert position by a block size

    Example:


        .. highlight:: python
        .. code-block:: python

            insert_position = [1, 4, 5, 2, 0, 3]
            unrolled_ip = unroll_insert_position(insert_position, 3)

        where ``unrolled_ip = [3, 4 ,5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 0, 1, 2, 9, 10, 11]``
    """
    unrolled_ip = np.repeat(insert_position, block_size) * block_size
    unrolled_ip += np.tile(np.arange(block_size), len(insert_position))
    return unrolled_ip

def create_periodic_mesh(mesh, indicator, mapping_function)-> tuple[dolfinx.mesh.Mesh, npt.NDArray[np.int32],
                                                                    npt.NDArray[np.int32]]:
    """
    Create a periodic mesh that takes all facets that satisfy the `indicator` function,
    and map the vertices of these facets to the vertices that satisfies the mapping function.

    Note:
        The cell ownership does not change, only additional ghosts are added to a given process

    Note:
        The vertex ownership does not change, only additional ghosts are added to a given process

    Returns:
        A tuple ``(new_mesh, replaced_vertices, replacement_map)`` where ``new_mesh`` is the new mesh with periodicity,
        ``replaced_vertices`` is a list of vertices of the input mesh that has been replaced (local to process).
        ``replacement_map`` is a map from the old vertices (local to process) to the new vertices (local to process).

        Note:
            This map does not contain additional ghost vertices added to the process that has taken over the facet or given away a facet.
    
    Example:

        .. code-block:: python
        .. highlight:: python
    
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 19)
        def indicator(x):
            return numpy.isclose(x[1], 1)

        def map(x):
            values = x.copy()
            values[1] -= 1
            return values

        periodic_mesh = create_periodic_mesh(mesh, indicator, map)        
    """    

    geometry = mesh.geometry._cpp_object
    topology = mesh.topology
    num_vertices = dolfinx.cpp.mesh.cell_num_vertices(mesh.topology.cell_type)
    num_nodes = mesh.geometry.dofmap.shape[1]


    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

    # Find facets through indicator function and incident vertices
    indicator_vertices = dolfinx.mesh.locate_entities_boundary(mesh, 0, indicator)

    # Communicate all vertices that are shared on all procs to all other procs
    vertex_map = mesh.topology.index_map(0)
    vector = dolfinx.la.vector(vertex_map, 1, dtype=np.int32)
    vector.array[:] = 0
    vector.array[indicator_vertices] = 1
    vector.scatter_reverse(dolfinx.la.InsertMode.add)
    vector.scatter_forward()
    indicator_vertices = np.flatnonzero(vector.array).astype(np.int32)

    num_owned_vertices = vertex_map.size_local    
    num_vertices_local = num_owned_vertices + mesh.topology.index_map(0).num_ghosts

    # Create first submap for vertices, where all indicated vertices are removed
    keep_vertices = np.ones(num_vertices_local, dtype=np.bool_)
    keep_vertices[indicator_vertices] = False
    reduced_vertices = np.flatnonzero(keep_vertices)
    sub_map_without_ghosts, sub_to_parent = dolfinx.cpp.common.create_sub_index_map(mesh.topology.index_map(0), reduced_vertices, allow_owner_change=False)

    # Compute reduced index map without indicator vertices
    num_vertices_local = mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts
    parent_to_sub = np.full(num_vertices_local, -1, dtype=np.int32)
    parent_to_sub[sub_to_parent] = np.arange(sub_to_parent.size, dtype=np.int32)

    if len(indicator_vertices) == 0:
        geom_coord = np.zeros((0,3), dtype=np.int32)
    else:
        geom_coord = dolfinx.mesh.entities_to_geometry(mesh, 0, indicator_vertices).reshape(-1)
    owned_vertex_coords = mesh.geometry.x[geom_coord]

    # Map vertices to new coordinates
    mapped_vertex_coords = mapping_function(owned_vertex_coords.T).T

    # Get vertices on process that has a cell colliding with point
    eps = 100*np.finfo(mesh.geometry.x.dtype).eps
    
    
    # For each vertex that will be replaced, find which process should take it over
    vertex_ownership_data = dolfinx.cpp.geometry.determine_point_ownership(mesh._cpp_object, mapped_vertex_coords, eps)

    # On process that has taken over a vertex, find the closest vertex (local to proc) that
    # will be its replacement
    acquired_vertex_coords = vertex_ownership_data.dest_points
    potential_closest_vertex = dolfinx.mesh.compute_incident_entities(mesh.topology, vertex_ownership_data.dest_cells, mesh.topology.dim, 0)
    closest_vertex_bb_tree = dolfinx.geometry.bb_tree(mesh,0, potential_closest_vertex)
    closest_vertex_mid_tree = dolfinx.geometry.create_midpoint_tree(mesh, 0, potential_closest_vertex)
    closest_vertex = dolfinx.geometry.compute_closest_entity(closest_vertex_bb_tree, closest_vertex_mid_tree, mesh, acquired_vertex_coords)
    
    # Map the closest vertex to its global index in the reduced submap
    global_vertices = sub_map_without_ghosts.local_to_global(parent_to_sub[closest_vertex]).astype(np.int64)
    original_mesh_vertex_owner = np.full(num_vertices_local, mesh.comm.rank, dtype=np.int32)
    original_mesh_vertex_owner[num_owned_vertices:] = vertex_map.owners
    send_vertex_owner = original_mesh_vertex_owner[parent_to_sub[closest_vertex]].copy()
  
    # For each vertex that is replaced, find the cells that are incident to the facet
    org_mesh_ext_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    mesh.topology.create_connectivity(0, mesh.topology.dim-1)

  
    # Get vertex and geometry dofs to send
    geom_dm = mesh.geometry.dofmap
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    
    # Pack data from process taking over vertex to process that has lost vertex

    assert np.all(vertex_ownership_data.dest_owners[:-1] <= vertex_ownership_data.dest_owners[1:]), "Vertex owners are not sorted"
    vertex_destinations, send_vertices_per_proc  = np.unique(vertex_ownership_data.dest_owners, return_counts=True)
    num_cells_per_proc = np.zeros_like(send_vertices_per_proc, dtype=np.int32) # Each vertex might be connected to multiple cells
    # Packing offset
    offsets = np.zeros(len(send_vertices_per_proc)+1, dtype=np.int32)
    np.cumsum(send_vertices_per_proc, out=offsets[1:])

    send_ghost_cells_from_new_owner = []
    new_cell_topology_dm = []
    # For each replacement vertex, find all facets that are connected to the vertex and is on the boundary.
    # Then, get each cell connected to this facet.
    for i in range(len(send_vertices_per_proc)):
        _vertices = closest_vertex[offsets[i]:offsets[i+1]] # Works because dest owners are sorted
        connected_facets = dolfinx.mesh.compute_incident_entities(mesh.topology, _vertices, 0, mesh.topology.dim-1)
        con_ext_facets = np.intersect1d(connected_facets, org_mesh_ext_facets)
        con_ext_cells = dolfinx.mesh.compute_incident_entities(mesh.topology, con_ext_facets, mesh.topology.dim-1, mesh.topology.dim)
        for cell in con_ext_cells:
            send_ghost_cells_from_new_owner.append(cell)
            num_cells_per_proc[i] += 1
            new_cell_topology_dm.extend(c_to_v.links(cell))

    new_cell_topology_dm = np.asarray(new_cell_topology_dm, dtype=np.int32)

    # Create new owner to old owner communicator
    vertex_sources, recv_vertices_per_proc = np.unique(vertex_ownership_data.src_owner, return_counts=True)
    new_owner_to_old_comm = mesh.comm.Create_dist_graph_adjacent(vertex_sources, vertex_destinations, reorder=False)    

    # Send replacement vertices to process that has lost vertex
    recv_replacement_vertices = np.empty(recv_vertices_per_proc.sum(), dtype=np.int64)
    all_to_allv(new_owner_to_old_comm, global_vertices, send_vertices_per_proc, recv_replacement_vertices, recv_vertices_per_proc)

    # Send owner of said vertex to the process that will use it as a replacement
    recv_replacement_owner = np.empty(recv_vertices_per_proc.sum(), dtype=np.int32)        
    all_to_allv(new_owner_to_old_comm, send_vertex_owner, send_vertices_per_proc, recv_replacement_owner, recv_vertices_per_proc)

    # For the data that will be received, the received has to be
    # ordered by their initial position in `mapped_vertex_coords`, not by src_rank
    current_rank_to_recv = compute_insert_position(vertex_ownership_data.src_owner, vertex_sources, recv_vertices_per_proc)
    # Invert map so that we can insert the data
    dest_ranks_to_current = np.zeros(mapped_vertex_coords.shape[0], dtype=np.int64)
    dest_ranks_to_current[current_rank_to_recv] = np.arange(len(dest_ranks_to_current), dtype=np.int32)

    # Global replacement index
    global_replacement_vertex = np.zeros(mapped_vertex_coords.shape[0], dtype=np.int64)
    global_replacement_vertex[dest_ranks_to_current] = recv_replacement_vertices
    global_replacement_owner = np.zeros(mapped_vertex_coords.shape[0], dtype=np.int64)
    global_replacement_owner[dest_ranks_to_current] = recv_replacement_owner

    # Set up ownership structure of cells, nodes and vertices on the process
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    cell_owners = get_ownership(cell_map)
    vertex_owners = get_ownership(sub_map_without_ghosts)

    # Map to global indices
    global_ghost_cells_from_new_owner = cell_map.local_to_global(np.array(send_ghost_cells_from_new_owner, dtype=np.int32)).astype(np.int64)
    subdofmap_for_new_owner_ghost_cells_local = parent_to_sub[new_cell_topology_dm.reshape(-1)]
    gl_new_cell_topology_dm = sub_map_without_ghosts.local_to_global(subdofmap_for_new_owner_ghost_cells_local).astype(np.int64)
    gl_new_cell_topology_owners = vertex_owners[subdofmap_for_new_owner_ghost_cells_local]

    # Compute number of cells to send and receive
    recv_num_cells = np.zeros_like(recv_vertices_per_proc, dtype=np.int32)
    new_owner_to_old_comm.Neighbor_alltoall(num_cells_per_proc, recv_num_cells)

    # Send cells and owners to process that lost vertex
    recv_potential_ghost_cells = np.empty(recv_num_cells.sum(), dtype=np.int64)
    all_to_allv(new_owner_to_old_comm, global_ghost_cells_from_new_owner, num_cells_per_proc, recv_potential_ghost_cells, recv_num_cells)

    recv_potential_cell_owners = np.empty(recv_num_cells.sum(), dtype=np.int32)
    send_ghost_cell_owners = cell_owners[send_ghost_cells_from_new_owner].copy()
    all_to_allv(new_owner_to_old_comm, send_ghost_cell_owners, num_cells_per_proc, recv_potential_cell_owners, recv_num_cells)

    # Send oci
    recv_potential_cell_oci = np.empty(recv_num_cells.sum(), dtype=np.int64)
    send_cell_oci = mesh.topology.original_cell_index[send_ghost_cells_from_new_owner].copy().astype(np.int64)
    all_to_allv(new_owner_to_old_comm, send_cell_oci, num_cells_per_proc, recv_potential_cell_oci, recv_num_cells)

    # Check if received cells are already in cell map
    potential_ghosts_as_local = cell_map.global_to_local(recv_potential_ghost_cells)
    cell_filter = np.flatnonzero(potential_ghosts_as_local == -1)
    new_cells_from_new_vertex_owner, vertex_owner_cell_position = np.unique(recv_potential_ghost_cells[cell_filter], return_index=True)    

    # Send dofmaps for topology
    new_top_dm_on_proc = np.empty((recv_num_cells.sum(), num_vertices), dtype=np.int64)
    all_to_allv(new_owner_to_old_comm, gl_new_cell_topology_dm, num_vertices*num_cells_per_proc,
                new_top_dm_on_proc, num_vertices*recv_num_cells)


    # Send ownership of vertices
    top_dm_ownership = np.empty_like(new_top_dm_on_proc, dtype=np.int32)
    all_to_allv(new_owner_to_old_comm,gl_new_cell_topology_owners, num_vertices*num_cells_per_proc,
                 top_dm_ownership, num_vertices*recv_num_cells)

    # Compute the vertex ghosts
    filtered_top_dm = new_top_dm_on_proc[cell_filter]
    local_dm = sub_map_without_ghosts.global_to_local(filtered_top_dm.reshape(-1))
    new_vertex_indicator = local_dm == -1
    shared_facet_vertices = filtered_top_dm.reshape(-1)[new_vertex_indicator]
    new_ghost_vertices, pos, inverse_map = np.unique(shared_facet_vertices, return_index=True, return_inverse=True)    
    new_ghost_owners = top_dm_ownership[cell_filter].reshape(-1)[new_vertex_indicator][pos]
    new_local_size = int(sub_map_without_ghosts.size_local)
    new_ghost_pos = new_local_size + sub_map_without_ghosts.num_ghosts
    local_ghost_indexing = new_ghost_pos + np.arange(len(new_ghost_vertices))
    local_dm[new_vertex_indicator] = local_ghost_indexing[inverse_map]
    new_ghosts = np.hstack([sub_map_without_ghosts.ghosts,new_ghost_vertices]).astype(np.int64)
    new_owners = np.hstack([sub_map_without_ghosts.owners, new_ghost_owners]).astype(np.int32)
    assert (new_owners != mesh.comm.rank).all()

    # Check if index is already in (reduced) vertex map
    local_replacement_vertex = sub_map_without_ghosts.global_to_local(global_replacement_vertex)
    is_local_indicator = local_replacement_vertex != -1
    existing_vertices = np.flatnonzero(is_local_indicator)

    # Vertex map is temporary, as we need to extend it with additional ghosts on the process taking over facets
    tmp_vertex_map = dolfinx.common.IndexMap(mesh.comm, new_local_size,  new_ghosts, new_owners)
    tmp_vertex_ownership = get_ownership(tmp_vertex_map)

    # Create replacement map
    replacement_map = parent_to_sub.copy()
    # Replace existing vertices
    replacement_map[indicator_vertices[existing_vertices]] = local_replacement_vertex[existing_vertices]

    # For new ghosts, add the to replacement map
    is_new_replacement = np.invert(is_local_indicator)
    replacement_ghosts = global_replacement_vertex[is_new_replacement]
    assert np.isin(replacement_ghosts, new_ghosts).all(), "Replacement ghost not in new ghost list"
    if len(replacement_ghosts) > 0:
        local_replacement_position = (new_ghosts==replacement_ghosts[:, None]).argmax(1)
        replacement_map[indicator_vertices[is_new_replacement]] = new_local_size + local_replacement_position

    geom_im = mesh.geometry.index_map()
    node_owners = get_ownership(geom_im)
    
    # Convert old vertex_to_dofmap to reduced set
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    new_c = replacement_map[c_to_v.array].reshape(-1, num_vertices)
    extra_dm = local_dm.reshape(-1, num_vertices)[cell_filter][vertex_owner_cell_position]

    # --- 2 --- Update geometry with new (ghosted) cells

    # Extend geometry with extra cells
    new_cell_geom_dm = geom_dm[send_ghost_cells_from_new_owner]
    gl_new_cell_geom_dm = geom_im.local_to_global(new_cell_geom_dm.reshape(-1)).astype(np.int64)

    # Send potential new ghosts
    add_geom_dm = np.empty((recv_num_cells.sum(), num_nodes), dtype=np.int64)
    all_to_allv(new_owner_to_old_comm, gl_new_cell_geom_dm, num_nodes*num_cells_per_proc, add_geom_dm, num_nodes*recv_num_cells)

    # Send ownersgpos of potential new ghost nodes
    send_geom_owners = node_owners[new_cell_geom_dm.reshape(-1)]
    add_geom_own = np.empty((recv_num_cells.sum(), num_nodes), dtype=np.int32)
    all_to_allv(new_owner_to_old_comm, send_geom_owners, num_nodes*num_cells_per_proc, add_geom_own, num_nodes*recv_num_cells)

    # Send igi for potential new nodes
    send_igi = mesh.geometry.input_global_indices[new_cell_geom_dm.reshape(-1)].astype(np.int64)
    recv_igi = np.empty((recv_num_cells.sum(), num_nodes), dtype=np.int64)
    all_to_allv(new_owner_to_old_comm, send_igi,num_nodes*num_cells_per_proc, 
                recv_igi,  num_nodes*recv_num_cells)


    # Compute new ghost nodes
    filtered_new_geometry_dm = add_geom_dm[cell_filter].flatten()
    filtered_new_geometry_owners = add_geom_own[cell_filter].flatten()
    igi_from_new_owner_on_subset = recv_igi[cell_filter].flatten()
    assert len(filtered_new_geometry_dm)==len(filtered_new_geometry_owners)
    local_geometry_dm = geom_im.global_to_local(filtered_new_geometry_dm)
    new_local_nodes = np.flatnonzero(local_geometry_dm == -1)
    new_ghost_nodes, gpos, ginverse_map = np.unique(filtered_new_geometry_dm[new_local_nodes], 
                                                    return_index=True, return_inverse=True)    

    new_igi = igi_from_new_owner_on_subset[new_local_nodes][gpos]
    new_ghost_owners = filtered_new_geometry_owners[new_local_nodes][gpos]
    num_local_nodes = geom_im.size_local
    new_node_pos = num_local_nodes + geom_im.num_ghosts
    local_geometry_dm[new_local_nodes] = (new_node_pos + np.arange(len(new_ghost_nodes),dtype=np.int32))[ginverse_map]

    # Communicate geometry coordinates (to process that has lost vertex)
    node_coordinates = mesh.geometry.x[new_cell_geom_dm.reshape(-1)].flatten()
    geom_coords = np.empty(num_nodes*3*recv_num_cells.sum(), dtype=mesh.geometry.x.dtype)
    send_coord_msg = [node_coordinates, num_nodes*3*num_cells_per_proc, mpi_dtype[mesh.geometry.x.dtype.type]]
    recv_coord_msg = [geom_coords, num_nodes*3*recv_num_cells, mpi_dtype[mesh.geometry.x.dtype.type]]
    new_owner_to_old_comm.Neighbor_alltoallv(send_coord_msg, recv_coord_msg)
    extra_geom_dm = local_geometry_dm.reshape(-1, num_nodes)[vertex_owner_cell_position]
   
    # --- 3 --- Communicate cells from process that has lost vertex to process that has taken over vertex

    # Pack additional cells to send from process losing facets (by midpoint) to the new owner
    # Given each indicator facet, find what process that owns the cell with the midpoint of the mapped midpoint
    indicator_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, indicator)
    # Communicate all vertices that are shared on all procs to all other procs
    facet_map = mesh.topology.index_map(mesh.topology.dim-1)
    fvector = dolfinx.la.vector(facet_map, 1, dtype=np.int32)
    fvector.array[:] = 0
    fvector.array[indicator_facets] = 1
    fvector.scatter_reverse(dolfinx.la.InsertMode.add)
    fvector.scatter_forward()
    indicator_facets = np.flatnonzero(fvector.array).astype(np.int32)


    facet_midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim-1, indicator_facets)
    mapping_facet_midpoints = mapping_function(facet_midpoints.T).T
    eps = 1000*np.finfo(mesh.geometry.x.dtype).eps
    mapped_midpoint_owner = dolfinx.cpp.geometry.determine_point_ownership(mesh._cpp_object, mapping_facet_midpoints, eps)
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    f_to_c = mesh.topology.connectivity(mesh.topology.dim-1, mesh.topology.dim)
    cells_losing_vertex = f_to_c.array[f_to_c.offsets[indicator_facets]]
    cells_losing_vertex_gl = cell_map.local_to_global(cells_losing_vertex)

    # Pack dofmap for each of these cells, replacing the vertices that are removed with mapped vertices
    renumbered_dm = new_c[cells_losing_vertex].reshape(-1)
    lost_cells_dm_global = tmp_vertex_map.local_to_global(renumbered_dm)
    lost_cells_dm_owners = tmp_vertex_ownership[renumbered_dm]

    # Pack dofmap,owners and igi of geometry, not in sorted by communication proc
    org_geom_dm_cells_losing_vertex = mesh.geometry.dofmap[cells_losing_vertex].reshape(-1)
    lost_geom_dm = geom_im.local_to_global(org_geom_dm_cells_losing_vertex)
    lost_geom_owner = node_owners[org_geom_dm_cells_losing_vertex]
    lost_geom_igi = mesh.geometry.input_global_indices[org_geom_dm_cells_losing_vertex]

    # Compute insertion position based on cell ownership
    lost_src_ranks, num_send_lost_cells = np.unique(mapped_midpoint_owner.src_owner, return_counts=True)
    lost_cell_insert_pos = compute_insert_position(mapped_midpoint_owner.src_owner, lost_src_ranks, num_send_lost_cells)
 
    # Pack cells data
    lost_cells_send_buffer = np.empty(len(cells_losing_vertex), dtype=np.int64)
    lost_cells_send_buffer[lost_cell_insert_pos] = cells_losing_vertex_gl
    lost_owners_send_buffer = np.empty_like(lost_cells_send_buffer, dtype=np.int32)
    lost_owners_send_buffer[lost_cell_insert_pos] = cell_owners[cells_losing_vertex]
    lost_oci_send_buffer = np.empty_like(lost_cells_send_buffer, dtype=np.int64)
    lost_oci_send_buffer[lost_cell_insert_pos] = mesh.topology.original_cell_index[cells_losing_vertex]

    # Pack topology data
    lost_insert_pos_top_dm = unroll_insert_position(lost_cell_insert_pos, num_vertices)
    lost_cells_dofmap_send_buffer = np.empty_like(lost_insert_pos_top_dm, dtype=np.int64)
    lost_cells_dofmap_send_buffer[lost_insert_pos_top_dm] = lost_cells_dm_global
    lost_cells_dofmap_owners_buffer = np.empty_like(lost_insert_pos_top_dm, dtype=np.int32)
    lost_cells_dofmap_owners_buffer[lost_insert_pos_top_dm] = lost_cells_dm_owners


    lost_insert_pos_geom_dm = unroll_insert_position(lost_cell_insert_pos, num_nodes)
    lost_cells_gdofmap_send_buffer = np.empty_like(lost_insert_pos_geom_dm, dtype=np.int64)
    lost_cells_gdofmap_send_buffer[lost_insert_pos_geom_dm] =  lost_geom_dm
    lost_cells_gdofmap_owner_buffer = np.empty_like(lost_insert_pos_geom_dm, dtype=np.int32)
    lost_cells_gdofmap_owner_buffer[lost_insert_pos_geom_dm] = lost_geom_owner
    lost_cells_gdofmap_igi_buffer =np.empty_like(lost_insert_pos_geom_dm, dtype=np.int64)
    lost_cells_gdofmap_igi_buffer[lost_insert_pos_geom_dm] = lost_geom_igi

    xtype = mesh.geometry.x.dtype
    lost_insert_pos_geom_coord = unroll_insert_position(lost_cell_insert_pos, 3*num_nodes)
    lost_geom_coords = mesh.geometry.x[org_geom_dm_cells_losing_vertex].flatten()
    lost_cells_coords_buffer = np.empty_like(lost_insert_pos_geom_coord, dtype=xtype)
    lost_cells_coords_buffer[lost_insert_pos_geom_coord] = lost_geom_coords

    # Create communicator
    recv_lost_cells_ranks, num_recv_lost_cells = np.unique(mapped_midpoint_owner.dest_owners, return_counts=True)
    lost_cells_to_gainer_comm = mesh.comm.Create_dist_graph_adjacent(
         recv_lost_cells_ranks.tolist(), lost_src_ranks.tolist(), reorder=False
    )


    total_recv_lost_cells = num_recv_lost_cells.sum()
    lost_cells_recv_buffer = np.empty(total_recv_lost_cells, dtype=np.int64)

    # Communicate cells
    all_to_allv(lost_cells_to_gainer_comm, lost_cells_send_buffer, num_send_lost_cells, lost_cells_recv_buffer, num_recv_lost_cells)

    # Communicate owners of potential new ghost cells
    lost_cells_owners_recv_buffer = np.empty_like(lost_cells_recv_buffer, dtype=np.int32)
    all_to_allv(lost_cells_to_gainer_comm, lost_owners_send_buffer, num_send_lost_cells, lost_cells_owners_recv_buffer, num_recv_lost_cells)

    # Communicate oci of potential new ghost cells
    lost_cells_oci_recv_buffer = np.empty_like(lost_cells_recv_buffer, dtype=np.int64)
    all_to_allv(lost_cells_to_gainer_comm, lost_oci_send_buffer, num_send_lost_cells, lost_cells_oci_recv_buffer, num_recv_lost_cells)

    # Communicate dofmap and ownership info
    lost_cells_dm_recv_buffer = np.empty((total_recv_lost_cells,num_vertices), dtype=np.int64)
    all_to_allv(lost_cells_to_gainer_comm, lost_cells_dofmap_send_buffer, num_send_lost_cells*num_vertices, lost_cells_dm_recv_buffer, num_recv_lost_cells*num_vertices)
    lost_cells_dm_owner_recv_buffer = np.empty_like(lost_cells_dm_recv_buffer, dtype=np.int32)
    all_to_allv(lost_cells_to_gainer_comm, lost_cells_dofmap_owners_buffer, num_send_lost_cells*num_vertices, lost_cells_dm_owner_recv_buffer, num_recv_lost_cells*num_vertices)


    # Communicate geometry dofmap, igi, owners and coordinates
    lost_cells_gdofmap_recv_buffer = np.empty((total_recv_lost_cells,num_nodes), dtype=np.int64)
    all_to_allv(lost_cells_to_gainer_comm, lost_cells_gdofmap_send_buffer, num_send_lost_cells*num_nodes, lost_cells_gdofmap_recv_buffer, num_recv_lost_cells*num_nodes)
    lost_cells_gdofmap_owner_recv_buffer = np.empty_like(lost_cells_gdofmap_recv_buffer, dtype=np.int32)
    all_to_allv(lost_cells_to_gainer_comm, lost_cells_gdofmap_owner_buffer, num_send_lost_cells*num_nodes, lost_cells_gdofmap_owner_recv_buffer, num_recv_lost_cells*num_nodes)
    lost_cells_igi_recv_buffer = np.empty_like(lost_cells_gdofmap_recv_buffer, dtype=np.int64)
    all_to_allv(lost_cells_to_gainer_comm, lost_cells_gdofmap_igi_buffer, num_send_lost_cells*num_nodes, lost_cells_igi_recv_buffer, num_recv_lost_cells*num_nodes)

    lost_cells_node_coords_recv_buffer = np.empty((total_recv_lost_cells, num_nodes, 3), dtype=mesh.geometry.x.dtype)   
    all_to_allv(lost_cells_to_gainer_comm, lost_cells_coords_buffer, num_send_lost_cells*num_nodes*3, lost_cells_node_coords_recv_buffer, num_recv_lost_cells*num_nodes*3)

    # Only add cells that are new on the process and only add them once
    new_lost_cells_indicator = np.flatnonzero(cell_map.global_to_local(lost_cells_recv_buffer) == -1)
    unique_lost_cells, unique_lost_cells_position = np.unique(lost_cells_recv_buffer[new_lost_cells_indicator], return_index=True) 
    unique_lost_cells_owners = lost_cells_owners_recv_buffer[new_lost_cells_indicator][unique_lost_cells_position]
    unique_lost_cells_oci = lost_cells_oci_recv_buffer[new_lost_cells_indicator][unique_lost_cells_position]
    # Get dofmap in local indices
    unique_lost_cells_dm = lost_cells_dm_recv_buffer[new_lost_cells_indicator][unique_lost_cells_position].reshape(-1)
    unique_lost_cells_dm_owners =  lost_cells_dm_owner_recv_buffer[new_lost_cells_indicator][unique_lost_cells_position].reshape(-1)
    lost_cells_dofs_as_local = tmp_vertex_map.global_to_local(unique_lost_cells_dm)

    # Find those vertex dofs that are new, and compute their new local vertex number
    lost_cells_new_vertices = np.flatnonzero(lost_cells_dofs_as_local == -1)
    lost_cells_unique_new_ghosts, unique_ghosts_position, unique_ghosts_to_new_vertices = np.unique(unique_lost_cells_dm[lost_cells_new_vertices], return_index=True, return_inverse=True)    
    lost_cells_ghost_owners = unique_lost_cells_dm_owners[lost_cells_new_vertices][unique_ghosts_position]
    lost_cells_ghost_insert_position = tmp_vertex_map.size_local + tmp_vertex_map.num_ghosts
    lost_cells_dofs_as_local[lost_cells_new_vertices] = (lost_cells_ghost_insert_position + np.arange(len(lost_cells_unique_new_ghosts),dtype=np.int32))[unique_ghosts_to_new_vertices]
    assert len(np.intersect1d(tmp_vertex_map.ghosts, lost_cells_unique_new_ghosts)) == 0

    # Compute ghost nodes for cells that are sent from process losing a facet
    filtered_geometry_dm = lost_cells_gdofmap_recv_buffer[new_lost_cells_indicator][unique_lost_cells_position].flatten()
    ext_geometry_dm = geom_im.global_to_local(filtered_geometry_dm)
    new_ext_nodes = np.flatnonzero(ext_geometry_dm == -1)
    ext_gm_ghosts, extg_pos, extg_inverse_map = np.unique(filtered_geometry_dm[new_ext_nodes], return_index=True, return_inverse=True)
    filtered_geometry_o = lost_cells_gdofmap_owner_recv_buffer[new_lost_cells_indicator][unique_lost_cells_position].flatten()
    ext_ghost_owners = filtered_geometry_o[extg_pos]
    ext_node_pos = num_local_nodes + geom_im.num_ghosts+ len(new_ghost_nodes)
    ext_geometry_dm[new_ext_nodes] = (ext_node_pos + np.arange(len(ext_gm_ghosts), dtype=np.int32))[extg_inverse_map]
    ext_geometry_dm = ext_geometry_dm.reshape(-1, num_nodes)
    filtered_geometry_coords = lost_cells_node_coords_recv_buffer[new_lost_cells_indicator][unique_lost_cells_position].reshape(-1, 3)[extg_pos]
    filtered_geometry_igi = lost_cells_igi_recv_buffer[new_lost_cells_indicator][unique_lost_cells_position].flatten()[extg_pos]

    # --- 4 --- Convert extended topology global dofmap into local dofmap
    assert len(np.intersect1d(recv_potential_ghost_cells[cell_filter][vertex_owner_cell_position], unique_lost_cells)) == 0, "Ghost in both additional maps"
    all_cell_ghosts = np.hstack([cell_map.ghosts, new_cells_from_new_vertex_owner ,unique_lost_cells]).astype(np.int64)
    all_cell_owners = np.hstack([cell_map.owners, recv_potential_cell_owners[cell_filter][vertex_owner_cell_position],  unique_lost_cells_owners]).astype(np.int32)
    assert (all_cell_owners != mesh.comm.rank).all(), "Ghosted cells on owned process"

    all_cell_oci = np.hstack([mesh.topology.original_cell_index, recv_potential_cell_oci[cell_filter][vertex_owner_cell_position], unique_lost_cells_oci]).astype(np.int64)

    all_ghosts = np.hstack([tmp_vertex_map.ghosts, lost_cells_unique_new_ghosts]).astype(np.int64)
    all_owners = np.hstack([tmp_vertex_map.owners, lost_cells_ghost_owners]).astype(np.int32)

    assert (all_owners != mesh.comm.rank).all(), "Ghosted vertices on owned process"
   
    # Create new cell and vertex map
    new_cell_map = dolfinx.common.IndexMap(mesh.comm, cell_map.size_local,  all_cell_ghosts, all_cell_owners)
    new_vertex_map = dolfinx.common.IndexMap(mesh.comm, tmp_vertex_map.size_local, all_ghosts, all_owners)

    new_c_to_v = dolfinx.graph.adjacencylist(np.vstack([new_c, extra_dm, lost_cells_dofs_as_local.reshape(-1, num_vertices)]))
    new_v_to_v = dolfinx.graph.adjacencylist(np.arange(new_vertex_map.size_local+new_vertex_map.num_ghosts, dtype=np.int32))
    assert (new_c_to_v.array < new_vertex_map.size_local + new_vertex_map.num_ghosts).all(), "Cell to vertex map is out of bounds"

    topology = dolfinx.cpp.mesh.Topology(MPI.COMM_WORLD, mesh.topology.cell_type)
    topology.set_index_map(0, new_vertex_map)
    topology.set_index_map(mesh.topology.dim, new_cell_map)
    topology.set_connectivity(new_v_to_v, 0,0)
    topology.set_connectivity(new_c_to_v, mesh.topology.dim, 0)
    try:
        topology.original_cell_index = all_cell_oci
    except AttributeError:
        print("Could not set original cell index (not supported in this version of DOLFIN-X)")

    c_el = dolfinx.fem.coordinate_element(mesh._ufl_domain.ufl_coordinate_element().basix_element)


    # ranges = MPI.COMM_WORLD.allgather(tmp_vertex_map.local_range)
    # for ghost, owner in zip(all_ghosts, all_owners):
    #     assert (ranges[owner][0] <= ghost) & (ghost < ranges[owner][1]), f"{MPI.COMM_WORLD.rank} Ghost {ghost} is not range {ranges[owner]}"
    assert ((all_ghosts < tmp_vertex_map.local_range[0]) | (tmp_vertex_map.local_range[1]<=all_ghosts)).all(), "Ghost "
    assert (new_vertex_map.ghosts<new_vertex_map.size_global).all(), "Ghosts larger than global size"
   
    # Create combined geometry
    extended_geom_ghosts = np.hstack([geom_im.ghosts, new_ghost_nodes, ext_gm_ghosts]).astype(np.int64)
    extended_geom_owners = np.hstack([geom_im.owners, new_ghost_owners, ext_ghost_owners]).astype(np.int32)
    extra_node_coords = geom_coords.reshape(-1,  num_nodes,3)[cell_filter].reshape(-1, 3)[new_local_nodes][gpos]

    extended_dofmap = np.vstack([mesh.geometry.dofmap, extra_geom_dm,
    ext_geometry_dm]).astype(np.int32)
    extended_coords = np.vstack([mesh.geometry.x, extra_node_coords,
   filtered_geometry_coords ]).astype(mesh.geometry.x.dtype)[:, :mesh.geometry.dim]
    new_node_im = dolfinx.common.IndexMap(mesh.comm, num_local_nodes, extended_geom_ghosts, extended_geom_owners)
    extended_igi = np.hstack([mesh.geometry.input_global_indices,  new_igi,
    filtered_geometry_igi]).astype(np.int64)
   
    geometry = dolfinx.mesh.create_geometry(new_node_im, extended_dofmap, c_el._cpp_object, extended_coords, extended_igi)
    if mesh.geometry.x.dtype == np.float64:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float64(mesh.comm, topology, geometry._cpp_object)
    elif mesh.geometry.x.dtype == np.float32:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float32(mesh.comm, topology, geometry._cpp_object)
    else:
        raise RuntimeError(f"Unsupported dtype for mesh {mesh.geometry.x.dtype}")
  

    new_mesh = dolfinx.mesh.Mesh(cpp_mesh, domain = ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element()))
    new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim)
    return new_mesh, indicator_vertices, replacement_map



if __name__ == "__main__":
    # N = 189
    # M = 123
    # N = 15
    # M = 10
    # mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, M,  ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    #                                        ,cell_type=dolfinx.mesh.CellType.quadrilateral)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    mesh, ct, ft =  dolfinx.io.gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD,  0, 2, partitioner=partitioner)

    dim = 1
    num_indices_local = mesh.topology.index_map(dim).size_local
    marker = np.arange(num_indices_local, dtype=np.int32)
    tags_old = dolfinx.mesh.meshtags(mesh, dim, marker, np.full_like(marker, MPI.COMM_WORLD.rank))




    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "org_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(tags_old, mesh.geometry)



    L_min = MPI.COMM_WORLD.allreduce(np.min(mesh.geometry.x[:,0]), op=MPI.MIN)
    L_max = MPI.COMM_WORLD.allreduce(np.max(mesh.geometry.x[:,0]), op=MPI.MAX)


    def indicator(x):
        return np.isclose(x[0], L_min)

    def mapping(x):
        values = x.copy()
        values[0] += L_max-L_min
        return values


    # mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    # old_num_exterior_facets = mesh.comm.allreduce(len(dolfinx.mesh.exterior_facet_indices(mesh.topology)), op=MPI.SUM)
    # assert old_num_exterior_facets == 2*N + 2*M, "Number of exterior facets is not correct"

    import time

    start = time.perf_counter()
    new_mesh, replaced_vertices, replacement_map = create_periodic_mesh(mesh, indicator, mapping)
    end = time.perf_counter()
    print(f"Create periodic mesh: {end-start:.3e}")

    if new_mesh.comm.size == 1:
        np.testing.assert_allclose(new_mesh.topology.original_cell_index, mesh.topology.original_cell_index)

    def num_vertices_per_entity(cell_type: dolfinx.mesh.CellType, dim:int)-> int:
        entity_vertices = dolfinx.cpp.mesh.get_entity_vertices(cell_type, dim)
        num_entity_vertices = entity_vertices.offsets[1:] - entity_vertices.offsets[:-1]

        assert np.unique(num_entity_vertices).size == 1, "Number of vertices per entity is not constant"
        return num_entity_vertices[0]


    # dim = 1
    # def marker_thing(x):
    #     return x[0]<= 1 + 1e-14

    # indices = dolfinx.mesh.locate_entities(mesh,  dim, marker_thing)
    # num_indices_local = mesh.topology.index_map(dim).size_local
    # local_indices = indices[indices < num_indices_local]
    # marker = np.arange(len(local_indices), dtype=np.int32)
    # tags_old = dolfinx.mesh.meshtags(mesh, dim, local_indices, marker)



    # with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "org_mesh.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(mesh)
    #     xdmf.write_meshtags(tags_old, mesh.geometry)


    tags_periodic = transfer_meshtags_to_periodic_mesh(mesh, new_mesh, replaced_vertices, ft)
    tags_periodic.name = "Periodic mesh tags"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "periodic_mesh_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(new_mesh)
        new_mesh.topology.create_connectivity(dim, new_mesh.topology.dim)    
        xdmf.write_meshtags(tags_periodic, new_mesh.geometry)


    # tags_periodic = transfer_meshtags_to_periodic_mesh(mesh, new_mesh, replaced_vertices, tags_old)
    # tags_periodic.name = "Periodic mesh tags"
    # with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "periodic_mesh_tags.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(new_mesh)
    #     new_mesh.topology.create_connectivity(dim, new_mesh.topology.dim)    
    #     xdmf.write_meshtags(tags_periodic, new_mesh.geometry)

    # new_mesh.topology.create_connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)
    # num_exterior_facets = mesh.comm.allreduce(len(dolfinx.mesh.exterior_facet_indices(new_mesh.topology)), op=MPI.SUM)
    #assert num_exterior_facets == 2*N, "Number of exterior facets is not correct"




    # Debug information
    # new_mesh.topology.create_connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)
    # f_to_c = new_mesh.topology.connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)
    # f_map = new_mesh.topology.index_map(new_mesh.topology.dim-1)
    # f_range = f_map.local_range


    # left_facets = dolfinx.mesh.locate_entities(new_mesh, new_mesh.topology.dim-1, indicator)
    # #owned_left_facets = left_facets[(f_range[0]<= left_facets) & (left_facets < f_range[1])]
    # lfm = dolfinx.mesh.compute_midpoints(new_mesh, new_mesh.topology.dim-1,left_facets)
    # for facet, midpoint in zip(left_facets, lfm):
    #     assert len(f_to_c.links(facet)) == 2, f"{MPI.COMM_WORLD.rank}: Left facet {facet} {midpoint} only connected to {f_to_c.links(facet)} cells"

    # new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)

    # right_facets = dolfinx.mesh.locate_entities(new_mesh, new_mesh.topology.dim-1,lambda x: np.isclose(x[0], 1.0))
    # owned_right_facets = right_facets[(f_range[0]<= right_facets) & (right_facets < f_range[1])]
    # rfm = dolfinx.mesh.compute_midpoints(new_mesh, new_mesh.topology.dim-1, owned_right_facets)
    # for facet, midpoint in zip(owned_right_facets, rfm):
    #     assert len(f_to_c.links(facet)) == 2, f"{MPI.COMM_WORLD.rank}: Left facet {facet} {midpoint} only connected to {f_to_c.links(facet)} cells"

    # new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)


    # with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "periodic_mesh.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(new_mesh)



    x = ufl.SpatialCoordinate(new_mesh)
    u_ex = ufl.sin(2*np.pi*x[0])
    h = 2 * ufl.Circumradius(new_mesh)
    h_avg = ufl.avg(h)
    gamma = dolfinx.fem.Constant(new_mesh, 100.)
    alpha = dolfinx.fem.Constant(new_mesh, 100.)

    V = dolfinx.fem.functionspace(new_mesh, ("DG", 2))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(new_mesh)
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += -ufl.inner(ufl.jump(v, n), ufl.avg(ufl.grad(u))) * ufl.dS
    F += -ufl.inner(ufl.avg(ufl.grad(v)), ufl.jump(u, n)) * ufl.dS
    F += +gamma / h_avg * ufl.inner(ufl.jump(v, n), ufl.jump(u, n)) * ufl.dS

    F += -ufl.inner(n, ufl.grad(u)) * v * ufl.ds

    F += -ufl.inner(n, ufl.grad(v)) * u * ufl.ds + alpha / h * ufl.inner(u, v) * ufl.ds
    F -= -ufl.inner(n, ufl.grad(v)) * u_ex * ufl.ds + alpha / h * ufl.inner(u_ex, v) * ufl.ds


    x = ufl.SpatialCoordinate(new_mesh)
    f = 100**x[0]*ufl.sin(0.5*np.pi * x[1])
    F-= ufl.inner(f, v) * ufl.dx
    a, L = ufl.system(F)
    import dolfinx.fem.petsc
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[], petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    uh = problem.solve()


    with dolfinx.io.VTXWriter(new_mesh.comm, "u_periodic.bp", [uh]) as writer:
        writer.write(0.0)


