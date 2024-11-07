# Create a periodic mesh in parallel
# SPDX-License-Identifier: MIT
# Author: Jørgen S. Dokken

from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import numpy.typing as npt



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


def create_periodic_mesh(mesh, indicator, mapping_function):
    """
    Create a periodic mesh that takes all facets that satisfy the `indicator` function,
    and map the vertices of these facets to the vertices that satisfies the mapping function.

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


    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

    # Find facets through indicator function and incident vertices
    indicator_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, indicator)
    indicator_vertices = dolfinx.mesh.compute_incident_entities(mesh.topology, indicator_facets, mesh.topology.dim -1, 0)

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
    vertex_owner = dolfinx.cpp.geometry.determine_point_ownership(mesh._cpp_object, mapped_vertex_coords, eps)
    assert np.all(vertex_owner.dest_owners[:-1] <= vertex_owner.dest_owners[1:]), "Vertex owners are not sorted"

    recv_coords = vertex_owner.dest_points
    recv_vertices =  dolfinx.mesh.compute_incident_entities(mesh.topology, vertex_owner.dest_cells, mesh.topology.dim, 0)
    bb_tree = dolfinx.geometry.bb_tree(mesh,0, recv_vertices)
    mid_tree = dolfinx.geometry.create_midpoint_tree(mesh, 0, recv_vertices)
    closest_vertex = dolfinx.geometry.compute_closest_entity(bb_tree, mid_tree, mesh, recv_coords)

    # Map closest vertex to global index
    vertex_map = mesh.topology.index_map(0)
    global_vertices = sub_map_without_ghosts.local_to_global(parent_to_sub[closest_vertex])

    vertex_sources, recv_vertices_per_proc = np.unique(vertex_owner.src_owner, return_counts=True)
    vertex_destinations, send_vertices_per_proc,  = np.unique(vertex_owner.dest_owners, return_counts=True)
    reverse_communicator = mesh.comm.Create_dist_graph_adjacent(vertex_sources, vertex_destinations, reorder=False)
    
    recv_vertices = np.empty(recv_vertices_per_proc.sum(), dtype=np.int64)
    send_msg = [global_vertices, send_vertices_per_proc, MPI.INT64_T]
    recv_msg = [recv_vertices, recv_vertices_per_proc, MPI.INT64_T]
    reverse_communicator.Neighbor_alltoallv(send_msg, recv_msg)

    # Send owner of said vertex to the process that will use it as a replacement
    recv_vertex_owner = np.empty(recv_vertices_per_proc.sum(), dtype=np.int32)
    owners = np.full(num_vertices_local, mesh.comm.rank, dtype=np.int32)
    owners[num_owned_vertices:] = vertex_map.owners
    send_msg = [owners[closest_vertex].copy(), send_vertices_per_proc, MPI.INT32_T]
    recv_msg = [recv_vertex_owner, recv_vertices_per_proc, MPI.INT32_T]
    reverse_communicator.Neighbor_alltoallv(send_msg, recv_msg)


    insert_position = compute_insert_position(vertex_owner.src_owner, vertex_sources, recv_vertices_per_proc)
    proc_to_vertex = np.zeros(mapped_vertex_coords.shape[0], dtype=np.int64)
    proc_to_vertex[insert_position] = np.arange(len(proc_to_vertex), dtype=np.int32)

    # Global replacement index
    global_replacement_vertex = np.zeros(mapped_vertex_coords.shape[0], dtype=np.int64)
    global_replacement_vertex[proc_to_vertex] = recv_vertices
    global_replacement_owner = np.zeros(mapped_vertex_coords.shape[0], dtype=np.int64)
    global_replacement_owner[proc_to_vertex] = recv_vertex_owner

    # For each vertex that is replaced, find the cels that are incident to the facet
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    mesh.topology.create_connectivity(0, mesh.topology.dim-1)
    num_cells_per_proc = np.zeros_like(send_vertices_per_proc, dtype=np.int32)
    new_ghost_cells = []
    new_cell_topology_dm = []
    offsets = np.zeros(len(send_vertices_per_proc)+1, dtype=np.int32)
    np.cumsum(send_vertices_per_proc, out=offsets[1:])

    # Set up ownership structure of cells, nodes and vertices on the process
    geom_im = mesh.geometry.index_map()

    cell_map = mesh.topology.index_map(mesh.topology.dim)
    cell_owners = get_ownership(cell_map)
    vertex_owners = get_ownership(sub_map_without_ghosts)

    # Get vertex and geometry dofs to send
    geom_dm = mesh.geometry.dofmap
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    for i in range(len(send_vertices_per_proc)):
        _vertices = closest_vertex[offsets[i]:offsets[i+1]]
        connected_facets = dolfinx.mesh.compute_incident_entities(mesh.topology, _vertices, 0, mesh.topology.dim-1)
        con_ext_facets = np.intersect1d(connected_facets, exterior_facets)
        con_ext_cells = dolfinx.mesh.compute_incident_entities(mesh.topology, con_ext_facets, mesh.topology.dim-1, mesh.topology.dim)
        for cell in con_ext_cells:
            new_ghost_cells.append(cell)
            num_cells_per_proc[i] += 1
            new_cell_topology_dm.extend(c_to_v.links(cell))


    new_cell_topology_dm = np.asarray(new_cell_topology_dm, dtype=np.int32)

    # Map to global indices
    gl_new_ghost_cells = cell_map.local_to_global(np.array(new_ghost_cells, dtype=np.int32))
    gl_new_cell_topology_dm = sub_map_without_ghosts.local_to_global(parent_to_sub[new_cell_topology_dm.reshape(-1)])
    
    cell_owners = cell_owners[new_ghost_cells]
    # Send ghost cells to process that has taken over vertex
    recv_num_cells = np.zeros_like(recv_vertices_per_proc, dtype=np.int32)
    reverse_communicator.Neighbor_alltoall(num_cells_per_proc, recv_num_cells)
    new_cells_on_proc = np.empty(recv_num_cells.sum(), dtype=np.int64)
    send_cells_msg = [gl_new_ghost_cells, num_cells_per_proc, MPI.INT64_T]
    recv_cells_msg = [new_cells_on_proc, recv_num_cells, MPI.INT64_T]
    reverse_communicator.Neighbor_alltoallv(send_cells_msg, recv_cells_msg)

    new_owners_on_proc = np.empty(recv_num_cells.sum(), dtype=np.int32)
    send_owner_msg = [cell_owners, num_cells_per_proc, MPI.INT32_T]
    recv_owner_msg = [new_owners_on_proc, recv_num_cells, MPI.INT32_T]
    reverse_communicator.Neighbor_alltoallv(send_owner_msg, recv_owner_msg)

    # Check if received cells are already in cell map
    potential_ghosts_as_local = cell_map.global_to_local(new_cells_on_proc)
    ghost_pos = np.flatnonzero(potential_ghosts_as_local == -1)

    # Send dofmaps for topology
    new_top_dm_on_proc = np.empty(num_vertices*recv_num_cells.sum(), dtype=np.int64)
    send_top_msg = [gl_new_cell_topology_dm, num_vertices*num_cells_per_proc, MPI.INT64_T]
    recv_top_msg = [new_top_dm_on_proc, num_vertices*recv_num_cells, MPI.INT64_T]
    reverse_communicator.Neighbor_alltoallv(send_top_msg, recv_top_msg)

    # Send ownership of vertices
    top_dm_ownership = np.empty(num_vertices*recv_num_cells.sum(), dtype=np.int32)
    send_top_omsg = [vertex_owners[parent_to_sub[new_cell_topology_dm.reshape(-1)]], num_vertices*num_cells_per_proc, MPI.INT32_T]
    recv_top_omsg = [top_dm_ownership, num_vertices*recv_num_cells, MPI.INT32_T]
    reverse_communicator.Neighbor_alltoallv(send_top_omsg, recv_top_omsg)

    # Compute the vertex ghosts
    local_dm = sub_map_without_ghosts.global_to_local(new_top_dm_on_proc)
    new_vertex_indicator = local_dm == -1
    shared_facet_vertices = new_top_dm_on_proc[new_vertex_indicator]
    new_ghost_vertices, pos, inverse_map = np.unique(shared_facet_vertices, return_index=True, return_inverse=True)    
    new_ghost_owners = top_dm_ownership[new_vertex_indicator][pos]
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

    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)

    # Pack additional cells to send from process losing facets (by midpoint) to the new owner
    # Given each indicator facet, find what process that owns the cell with the midpoint of the mapped midpoint
    facet_midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim-1, indicator_facets)
    mapping_facet_midpoints = mapping_function(facet_midpoints.T).T
    eps = 100*np.finfo(mesh.geometry.x.dtype).eps
    mapped_midpoint_owner = dolfinx.cpp.geometry.determine_point_ownership(mesh._cpp_object, mapping_facet_midpoints, eps)
    midpoint_cells = dolfinx.fem.compute_integration_domains(dolfinx.fem.IntegralType.exterior_facet, mesh.topology, indicator_facets, mesh.topology.dim-1)[::2]
    midpoint_cells_as_global = cell_map.local_to_global(midpoint_cells)
 
    # Pack dofmap for each of these cells, replacing the vertices that are removed with mapped vertices
    local_ghosted_map = c_to_v.array.copy().reshape(-1, num_vertices)[midpoint_cells]
    extended_map = replacement_map[local_ghosted_map].reshape(-1)
    global_extended = tmp_vertex_map.local_to_global(extended_map)

    tmp_vertex_ownership = get_ownership(tmp_vertex_map)
    ext_top_dm_owners = tmp_vertex_ownership[extended_map]
    assert (midpoint_cells < cell_map.size_local).all(), "Cell is not owned by process"

    # Pack geometry dofmap, owners, igi and coordinates
    num_nodes = mesh.geometry.dofmap.shape[1]
    geom_ext_dm = geom_im.local_to_global(mesh.geometry.dofmap[midpoint_cells].reshape(-1))
    node_owners = get_ownership(geom_im)
    geom_ext_owner = node_owners[mesh.geometry.dofmap[midpoint_cells].reshape(-1)]
    geom_ext_igi = mesh.geometry.input_global_indices[mesh.geometry.dofmap[midpoint_cells].reshape(-1)]
    geom_ext_coords = mesh.geometry.x[mesh.geometry.dofmap[midpoint_cells].reshape(-1)].flatten()

    # Compute insertion map
    ext_src_ranks, num_ext_send_cells = np.unique(mapped_midpoint_owner.src_owner, return_counts=True)
    insert_pos_ext_cells = compute_insert_position(mapped_midpoint_owner.src_owner, ext_src_ranks, num_ext_send_cells)
    insert_pos_ext_top_dm = unroll_insert_position(insert_pos_ext_cells, num_vertices)
    insert_pos_ext_geom_dm = unroll_insert_position(insert_pos_ext_cells, num_nodes)
    insert_pos_ext_geom_coord = unroll_insert_position(insert_pos_ext_cells, 3*num_nodes)

    # Pack data to send
    send_ext_cells = np.empty(len(midpoint_cells), dtype=np.int64)
    send_ext_cells[insert_pos_ext_cells] = midpoint_cells_as_global
    send_ext_dm = global_extended[insert_pos_ext_top_dm]
    send_ext_gm = geom_ext_dm[insert_pos_ext_geom_dm]
    send_ext_top_owners = ext_top_dm_owners[insert_pos_ext_top_dm]

    send_ext_gm_owners = geom_ext_owner[insert_pos_ext_geom_dm]
    send_ext_igi = geom_ext_igi[insert_pos_ext_geom_dm]
    send_ext_coords = geom_ext_coords[insert_pos_ext_geom_coord]

    # Create communicator
    ext_dest_ranks, num_ext_recv_cells = np.unique(mapped_midpoint_owner.dest_owners, return_counts=True)
    remove_to_owner_comm = mesh.comm.Create_dist_graph_adjacent(
         ext_dest_ranks.tolist(), ext_src_ranks.tolist(), reorder=False
    )

    # Communicate potential new ghost cells (shared facet)
    ext_cell_msg = [send_ext_cells, num_ext_send_cells, MPI.INT64_T]
    ext_recv_cells = [np.empty(num_ext_recv_cells.sum(), dtype=np.int64), num_ext_recv_cells, MPI.INT64_T]
    remove_to_owner_comm.Neighbor_alltoallv(ext_cell_msg, ext_recv_cells)

    # Communicate owners of potential new ghost cells
    ext_cello_msg = [np.full_like(send_ext_cells, mesh.comm.rank, dtype=np.int32), num_ext_send_cells, MPI.INT32_T]
    ext_recv_cowner = [np.empty(num_ext_recv_cells.sum(), dtype=np.int32), num_ext_recv_cells, MPI.INT32_T]
    remove_to_owner_comm.Neighbor_alltoallv(ext_cello_msg, ext_recv_cowner)

    # Communicate dofmap and ownership info
    ext_topdm_msg = [send_ext_dm, num_ext_send_cells*num_vertices, MPI.INT64_T]
    ext_recv_top_dm_msg = [np.empty(num_ext_recv_cells.sum()*num_vertices, dtype=np.int64), num_ext_recv_cells*num_vertices, MPI.INT64_T]
    remove_to_owner_comm.Neighbor_alltoallv(ext_topdm_msg, ext_recv_top_dm_msg)

    ext_topdmo_msg = [send_ext_top_owners, num_ext_send_cells*num_vertices, MPI.INT32_T]
    ext_recv_top_dmo_msg = [np.empty(num_ext_recv_cells.sum()*num_vertices, dtype=np.int32), num_ext_recv_cells*num_vertices, MPI.INT32_T]
    remove_to_owner_comm.Neighbor_alltoallv(ext_topdmo_msg, ext_recv_top_dmo_msg)

    # Communicate geometry dofmap, igi, owners and coordinates
    ext_geom_msg = [send_ext_gm, num_ext_send_cells*num_nodes, MPI.INT64_T]
    ext_recv_geom_msg = [np.empty(num_ext_recv_cells.sum()*num_nodes, dtype=np.int64), num_ext_recv_cells*num_nodes, MPI.INT64_T]
    remove_to_owner_comm.Neighbor_alltoallv(ext_geom_msg, ext_recv_geom_msg)

    ext_geomo_msg = [send_ext_gm_owners, num_ext_send_cells*num_nodes, MPI.INT32_T]
    ext_recv_geomo_msg = [np.empty(num_ext_recv_cells.sum()*num_nodes, dtype=np.int32), num_ext_recv_cells*num_nodes, MPI.INT32_T]
    remove_to_owner_comm.Neighbor_alltoallv(ext_geomo_msg, ext_recv_geomo_msg)

    recv_ext_igi = np.empty(num_nodes*num_ext_recv_cells.sum(), dtype=np.int64)
    send_ext_igi_msg = [send_ext_igi, num_nodes*num_ext_send_cells, MPI.INT64_T]
    recv_ext_igi_msg = [recv_ext_igi, num_nodes*num_ext_recv_cells, MPI.INT64_T]
    remove_to_owner_comm.Neighbor_alltoallv(send_ext_igi_msg, recv_ext_igi_msg)


    mpi_dtype = {np.float64: MPI.DOUBLE, np.float32: MPI.FLOAT}

    xdtype = mpi_dtype[mesh.geometry.x.dtype.type]
    recv_ext_coords = np.empty(3*num_nodes*num_ext_recv_cells.sum(), dtype=mesh.geometry.x.dtype)
    send_ext_coords_msg = [send_ext_coords, 3*num_nodes*num_ext_send_cells, xdtype]
    recv_ext_coords_msg = [recv_ext_coords, 3*num_nodes*num_ext_recv_cells, xdtype]
    remove_to_owner_comm.Neighbor_alltoallv(send_ext_coords_msg, recv_ext_coords_msg)

    # Create new cell map
    # Check if received cells are already in cell map
    recv_ext_ghosts = cell_map.global_to_local(ext_recv_cells[0])
    ext_ghost_pos = np.flatnonzero(recv_ext_ghosts == -1)

    all_cell_ghosts = np.hstack([cell_map.ghosts, new_cells_on_proc[ghost_pos], ext_recv_cells[0][ext_ghost_pos]]).astype(np.int64)
    all_cell_owners = np.hstack([cell_map.owners, new_owners_on_proc[ghost_pos], ext_recv_cowner[0][ext_ghost_pos]]).astype(np.int32)

    assert (all_cell_owners != mesh.comm.rank).all(), "Ghosted cells on owned process"
    new_cell_map = dolfinx.common.IndexMap(mesh.comm, cell_map.size_local,  all_cell_ghosts, all_cell_owners)

    # Convert extended topology global dofmap into local dofmap
    new_ext_cells_dm = ext_recv_top_dm_msg[0].reshape(-1, num_vertices)[ext_ghost_pos].reshape(-1)
    recv_ext_dm = tmp_vertex_map.global_to_local(new_ext_cells_dm)
    new_ext_vertices = np.flatnonzero(recv_ext_dm == -1)
    new_ext_ghosts, ext_gpos, ext_ginverse_map = np.unique(new_ext_cells_dm[new_ext_vertices], return_index=True, return_inverse=True)    
    new_ext_owners = ext_recv_top_dmo_msg[0][new_ext_vertices][ext_gpos]
    new_vertex_pos = tmp_vertex_map.size_local + tmp_vertex_map.num_ghosts
    recv_ext_dm[new_ext_vertices] = (new_vertex_pos + np.arange(len(new_ext_ghosts),dtype=np.int32))[ext_ginverse_map]
    all_ghosts = np.hstack([tmp_vertex_map.ghosts, new_ext_ghosts]).astype(np.int64)
    all_owners = np.hstack([tmp_vertex_map.owners, new_ext_owners]).astype(np.int32)

    assert (all_owners != mesh.comm.rank).all(), "Ghosted vertices on owned process"
    new_vertex_map = dolfinx.common.IndexMap(mesh.comm, tmp_vertex_map.size_local, all_ghosts, all_owners)


    # Convert old vertex_to_dofmap to reduced set
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    new_c = replacement_map[c_to_v.array].reshape(-1, num_vertices)
    extra_dm = local_dm.reshape(-1, num_vertices)[ghost_pos]

    new_c_to_v = dolfinx.graph.adjacencylist(np.vstack([new_c, extra_dm, recv_ext_dm.reshape(-1, num_vertices)]))
    new_v_to_v = dolfinx.graph.adjacencylist(np.arange(new_vertex_map.size_local+new_vertex_map.num_ghosts, dtype=np.int32))
    assert (new_c_to_v.array < new_vertex_map.size_local + new_vertex_map.num_ghosts).all(), "Cell to vertex map is out of bounds"

    topology = dolfinx.cpp.mesh.Topology(MPI.COMM_WORLD, mesh.topology.cell_type)
    topology.set_index_map(0, new_vertex_map)
    topology.set_index_map(mesh.topology.dim, new_cell_map)
    topology.set_connectivity(new_v_to_v, 0,0)
    topology.set_connectivity(new_c_to_v, mesh.topology.dim, 0)
    c_el = dolfinx.fem.coordinate_element(mesh._ufl_domain.ufl_coordinate_element().basix_element)

    # Extend geometry with extra cells
    new_cell_geom_dm = geom_dm[new_ghost_cells]
    gl_new_cell_geom_dm = geom_im.local_to_global(new_cell_geom_dm.reshape(-1))

    # Send potential new ghosts
    add_geom_dm = np.empty(num_nodes*recv_num_cells.sum(), dtype=np.int64)
    send_geom_msg = [gl_new_cell_geom_dm, num_nodes*num_cells_per_proc, MPI.INT64_T]
    recv_geom_msg = [add_geom_dm, num_nodes*recv_num_cells, MPI.INT64_T]
    reverse_communicator.Neighbor_alltoallv(send_geom_msg, recv_geom_msg)

    # Send owners of potential new ghost nodes
    send_geom_owners = node_owners[new_cell_geom_dm.reshape(-1)]
    add_geom_own = np.empty(num_nodes*recv_num_cells.sum(), dtype=np.int32)
    send_geom_msg = [send_geom_owners, num_nodes*num_cells_per_proc, MPI.INT32_T]
    recv_geom_msg = [add_geom_own, num_nodes*recv_num_cells, MPI.INT32_T]
    reverse_communicator.Neighbor_alltoallv(send_geom_msg, recv_geom_msg)

    # Send igi for potential new nodes
    send_igi = mesh.geometry.input_global_indices[new_cell_geom_dm.reshape(-1)]
    recv_igi = np.empty(num_nodes*recv_num_cells.sum(), dtype=np.int64)
    send_igi_msg = [send_igi, num_nodes*num_cells_per_proc, MPI.INT64_T]
    recv_igi_msg = [recv_igi, num_nodes*recv_num_cells, MPI.INT64_T]
    reverse_communicator.Neighbor_alltoallv(send_igi_msg, recv_igi_msg)

    # Compute new ghost nodes
    local_geometry_dm = geom_im.global_to_local(add_geom_dm)
    new_local_nodes = np.flatnonzero(local_geometry_dm == -1)
    new_ghost_nodes, gpos, ginverse_map = np.unique(add_geom_dm[new_local_nodes], return_index=True, return_inverse=True)    
    new_ghost_owners = add_geom_own[new_local_nodes][gpos]
    num_local_nodes = geom_im.size_local
    new_node_pos = num_local_nodes + geom_im.num_ghosts
    local_geometry_dm[new_local_nodes] = (new_node_pos + np.arange(len(new_ghost_nodes),dtype=np.int32))[ginverse_map]

    # Compute ghost nodes for cells that are sent from process losing a facet
    filtered_geometry_dm = ext_recv_geom_msg[0].reshape(-1, num_nodes)[ext_ghost_pos].flatten()
    ext_geometry_dm = geom_im.global_to_local(filtered_geometry_dm)
    new_ext_nodes = np.flatnonzero(ext_geometry_dm == -1)
    ext_gm_ghosts, extg_pos, extg_inverse_map = np.unique(filtered_geometry_dm[new_ext_nodes], return_index=True, return_inverse=True)
    filtered_geometry_o = ext_recv_geomo_msg[0].reshape(-1, num_nodes)[ext_ghost_pos].flatten()
    ext_ghost_owners = filtered_geometry_o[new_ext_nodes][extg_pos]
    ext_node_pos = num_local_nodes + geom_im.num_ghosts+ len(new_ghost_nodes)
    ext_geometry_dm[new_ext_nodes] = (ext_node_pos + np.arange(len(new_ext_nodes), dtype=np.int32))[extg_inverse_map]
    ext_geometry_dm = ext_geometry_dm.reshape(-1, num_nodes)
    filtered_coords = recv_ext_coords.reshape(-1, 3)[new_ext_nodes][extg_pos]
    ext_ghost_igi = recv_ext_igi[new_ext_nodes][extg_pos]

    # Communicate geometry coordiantes (to process that has lost vertex)
    node_coordinates = mesh.geometry.x[new_cell_geom_dm.reshape(-1)].flatten()
    geom_coords = np.empty(num_nodes*3*recv_num_cells.sum(), dtype=mesh.geometry.x.dtype)
    mpi_dtype = {np.float64: MPI.DOUBLE, np.float32: MPI.FLOAT}
    send_coord_msg = [node_coordinates, num_nodes*3*num_cells_per_proc, mpi_dtype[mesh.geometry.x.dtype.type]]
    recv_coord_msg = [geom_coords, num_nodes*3*recv_num_cells, mpi_dtype[mesh.geometry.x.dtype.type]]
    reverse_communicator.Neighbor_alltoallv(send_coord_msg, recv_coord_msg)
    extra_geom_dm = local_geometry_dm.reshape(-1, num_nodes)[ghost_pos]

    extended_geom_ghosts = np.hstack([geom_im.ghosts, new_ghost_nodes, ext_gm_ghosts]).astype(np.int64)
    extended_geom_owners = np.hstack([geom_im.owners, new_ghost_owners, ext_ghost_owners]).astype(np.int32)
    extra_node_coords = geom_coords.reshape(-1,  3)[new_local_nodes][gpos]

    extended_dofmap = np.vstack([mesh.geometry.dofmap, extra_geom_dm,
    ext_geometry_dm]).astype(np.int32)
    extended_coords = np.vstack([mesh.geometry.x, extra_node_coords,
   filtered_coords ]).astype(mesh.geometry.x.dtype)[:, :mesh.geometry.dim]
    new_node_im = dolfinx.common.IndexMap(mesh.comm, num_local_nodes, extended_geom_ghosts, extended_geom_owners)
    extended_igi = np.hstack([mesh.geometry.input_global_indices, recv_igi[new_local_nodes][gpos],
    ext_ghost_igi]).astype(np.int64)

    geometry = dolfinx.mesh.create_geometry(new_node_im, extended_dofmap, c_el._cpp_object, extended_coords, extended_igi)
    if mesh.geometry.x.dtype == np.float64:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float64(mesh.comm, topology, geometry._cpp_object)
    elif mesh.geometry.x.dtype == np.float32:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float32(mesh.comm, topology, geometry._cpp_object)
    else:
        raise RuntimeError(f"Unsupported dtype for mesh {mesh.geometry.x.dtype}")
  

    new_mesh = dolfinx.mesh.Mesh(cpp_mesh, domain = ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element()))

    return new_mesh



# mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 1, cell_type=dolfinx.mesh.CellType.quadrilateral)#, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 100, 100, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

mesh.topology.create_connectivity(0,2)
cell_marker = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)#np.arange(*mesh.topology.index_map(mesh.topology.dim).local_range , dtype=np.int32)
cell_ind = np.arange(len(cell_marker), dtype=np.int32)
ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cell_ind, cell_marker)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "org_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)

def indicator(x):
    return np.isclose(x[0], 0.0)

def mapping(x):
    values = x.copy()
    values[0] += 1
    return values

# mpirun -n 2 python3 script.py 
new_mesh = create_periodic_mesh(mesh, indicator, mapping)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "periodic_mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(new_mesh)



# new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)
# new_mesh.topology.create_connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)

# c_to_f_new = new_mesh.topology.connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)
# mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
# c_to_f = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim-1)
# new_mesh.topology.create_connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)
# f_to_c_new = new_mesh.topology.connectivity(new_mesh.topology.dim-1, new_mesh.topology.dim)
# new_mesh.topology.create_connectivity(new_mesh.topology.dim-1,0)
# f_to_v_new = new_mesh.topology.connectivity(new_mesh.topology.dim-1,0)
# if MPI.COMM_WORLD.rank == 1:
#     print(f_to_c_new.links(5), dolfinx.mesh.compute_midpoints(new_mesh, 1, np.array([5],dtype=np.int32)))
#     print(f_to_v_new.links(5))

# V_out = dolfinx.fem.functionspace(new_mesh, ("DG", 2, (new_mesh.geometry.dim, )))
# u_out = dolfinx.fem.Function(V_out)
# u_out.interpolate(lambda x:( np.sin(2*np.pi*x[0]), x[0]))

# with dolfinx.io.VTXWriter(new_mesh.comm, "u_periodic.bp", [u_out]) as writer:
#     writer.write(0.0)
# exit()

new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)

# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "periodic_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(new_mesh)



x = ufl.SpatialCoordinate(new_mesh)
u_ex = ufl.sin(2*np.pi*x[0])
h = 2 * ufl.Circumradius(new_mesh)
h_avg = ufl.avg(h)
gamma = dolfinx.fem.Constant(new_mesh, 10.)
alpha = dolfinx.fem.Constant(new_mesh, 10.)

V = dolfinx.fem.functionspace(new_mesh, ("DG", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
n = ufl.FacetNormal(new_mesh)
F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
F += -ufl.inner(ufl.jump(v, n), ufl.avg(ufl.grad(u))) * ufl.dS
F += -ufl.inner(ufl.avg(ufl.grad(v)), ufl.jump(u, n)) * ufl.dS
F += +gamma / h_avg * ufl.inner(ufl.jump(v, n), ufl.jump(u, n)) * ufl.dS

F += ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.inner(n, ufl.grad(u)) * v * ufl.ds

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

exit()
