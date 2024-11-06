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

    # Only work on owned vertices for mapping   

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
    vertex_owners = get_ownership(vertex_map)
    node_owners = get_ownership(geom_im)

    # Get vertex and geometry dofs to send
    geom_dm = mesh.geometry.dofmap
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    for i in range(len(send_vertices_per_proc)):
        _vertices = closest_vertex[offsets[i]:offsets[i+1]]
        connected_facets = dolfinx.mesh.compute_incident_entities(mesh.topology, _vertices, 0, mesh.topology.dim-1)
        con_ext_facets = np.intersect1d(connected_facets, exterior_facets)
        con_ext_cells = dolfinx.mesh.compute_incident_entities(mesh.topology, con_ext_facets, mesh.topology.dim-1, mesh.topology.dim)
        for  cell in con_ext_cells:
            # If cell is not owned by the process the vertex came from
            if cell_owners[cell] != vertex_owner.dest_cells[i]:
                new_ghost_cells.append(cell)
                num_cells_per_proc[i] += 1
                new_cell_topology_dm.extend(c_to_v.links(cell))

    new_cell_geom_dm = geom_dm[new_ghost_cells]
    new_cell_topology_dm = np.asarray(new_cell_topology_dm, dtype=np.int32)

    # Map to global indices
    gl_new_ghost_cells = cell_map.local_to_global(np.array(new_ghost_cells, dtype=np.int32))
    gl_new_cell_topology_dm = sub_map_without_ghosts.local_to_global(parent_to_sub[new_cell_topology_dm.reshape(-1)])
    gl_new_cell_geom_dm = geom_im.local_to_global(new_cell_geom_dm.reshape(-1))
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
    
    # Create new cell map
    all_cell_ghosts = np.hstack([cell_map.ghosts, new_cells_on_proc[ghost_pos]]).astype(np.int64)
    all_cell_owners = np.hstack([cell_map.owners, new_owners_on_proc[ghost_pos]]).astype(np.int32)
    new_cell_map = dolfinx.common.IndexMap(mesh.comm, cell_map.size_local,  all_cell_ghosts, all_cell_owners)

    # Send dofmaps for topology
    num_vertices = dolfinx.cpp.mesh.cell_num_vertices(mesh.topology.cell_type)
    new_top_dm_on_proc = np.empty(num_vertices*recv_num_cells.sum(), dtype=np.int64)
    send_top_msg = [gl_new_cell_topology_dm, num_vertices*num_cells_per_proc, MPI.INT64_T]
    recv_top_msg = [new_top_dm_on_proc, num_vertices*recv_num_cells, MPI.INT64_T]
    reverse_communicator.Neighbor_alltoallv(send_top_msg, recv_top_msg)

    # Send ownership of vertices
    top_dm_ownership = np.empty(num_vertices*recv_num_cells.sum(), dtype=np.int32)
    send_top_omsg = [vertex_owners[gl_new_cell_topology_dm], num_vertices*num_cells_per_proc, MPI.INT32_T]
    recv_top_omsg = [top_dm_ownership, num_vertices*recv_num_cells, MPI.INT32_T]
    reverse_communicator.Neighbor_alltoallv(send_top_omsg, recv_top_omsg)


    # Compute the vertex ghosts
    local_dm = sub_map_without_ghosts.global_to_local(new_top_dm_on_proc)
    shared_facet_vertices = new_top_dm_on_proc[local_dm == -1]
    new_ghost_vertices, pos, inverse_map = np.unique(shared_facet_vertices, return_index=True, return_inverse=True)    
    new_ghost_owners = top_dm_ownership[local_dm == -1][pos]
    new_local_size = int(sub_map_without_ghosts.size_local)
    new_ghost_pos = new_local_size + sub_map_without_ghosts.num_ghosts
    local_ghost_indexing = new_ghost_pos + np.arange(len(new_ghost_vertices))
    local_dm[local_dm == -1] = local_ghost_indexing[inverse_map]
    new_ghosts = np.hstack([sub_map_without_ghosts.ghosts,new_ghost_vertices]).astype(np.int64)
    new_owners = np.hstack([sub_map_without_ghosts.owners, new_ghost_owners]).astype(np.int32)


    # Check if index is already in (reduced) vertex map
    local_replacement_vertex = sub_map_without_ghosts.global_to_local(global_replacement_vertex)
    is_local_indicator = local_replacement_vertex != -1
    existing_vertices = np.flatnonzero(is_local_indicator)

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
        replacement_map[indicator_vertices[is_new_replacement]] =local_replacement_position

    
    new_vertex_map = dolfinx.common.IndexMap(mesh.comm, new_local_size,  new_ghosts, new_owners)

    #  num_nodes = mesh.geometry.dofmap.shape[1]
    # new_geom_dm_on_proc = np.empty(num_nodes*recv_num_cells.sum(), dtype=np.int64)
    # send_geom_msg = [gl_new_cell_geom_dm, num_nodes*num_cells_per_proc, MPI.INT64_T]
    # recv_geom_msg = [new_geom_dm_on_proc, num_nodes*recv_num_cells, MPI.INT64_T]
    # reverse_communicator.Neighbor_alltoallv(send_geom_msg, recv_geom_msg)

    #local_geom_dm = geom_im.global_to_local(new_geom_dm_on_proc)
    # new_local_geometry = new_geom_dm_on_proc[local_geom_dm  == -1]
    # new_node_owners = top_dm_ownership[local_dm == 1]
 
 
    # Convert old vertex_to_dofmap to reduced set
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    new_c = replacement_map[c_to_v.array]
    new_o = c_to_v.offsets.copy()
    new_c_to_v = dolfinx.graph.adjacencylist(new_c, new_o)
    new_v_to_v = dolfinx.graph.adjacencylist(np.arange(new_vertex_map.size_local+new_vertex_map.num_ghosts, dtype=np.int32))
    assert (new_c_to_v.array < new_vertex_map.size_local + new_vertex_map.num_ghosts).all(), "Cell to vertex map is out of bounds"
)

    # print(new_vertex_map.size_local, new_vertex_map.ghosts, new_vertex_map.owners, new_vertex_map.size_global, new_vertex_map.local_range)
    # exit()
    topology = dolfinx.cpp.mesh.Topology(MPI.COMM_WORLD, mesh.topology.cell_type)
    topology.set_index_map(0, new_vertex_map)
    topology.set_index_map(mesh.topology.dim, mesh.topology.index_map(mesh.topology.dim))
    topology.set_connectivity(new_v_to_v, 0,0)
    topology.set_connectivity(new_c_to_v, mesh.topology.dim, 0)
    c_el = dolfinx.fem.coordinate_element(mesh._ufl_domain.ufl_coordinate_element().basix_element)
    geometry = dolfinx.mesh.create_geometry(mesh.geometry.index_map(), mesh.geometry.dofmap, c_el._cpp_object, mesh.geometry.x[:, :mesh.geometry.dim].copy(),  mesh.geometry.input_global_indices)
    if mesh.geometry.x.dtype == np.float64:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float64(mesh.comm, topology, geometry._cpp_object)
    elif mesh.geometry.x.dtype == np.float32:
        cpp_mesh = dolfinx.cpp.mesh.Mesh_float32(mesh.comm, topology, geometry._cpp_object)
    else:
        raise RuntimeError(f"Unsupported dtype for mesh {mesh.geometry.x.dtype}")
  

    new_mesh = dolfinx.mesh.Mesh(cpp_mesh, domain = ufl.Mesh(mesh._ufl_domain.ufl_coordinate_element()))

    return new_mesh



mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 26, 13)


def indicator(x):
    return np.isclose(x[0], 0.0)

def mapping(x):
    values = x.copy()
    values[0] += 1
    return values

# mpirun -n 2 python3 script.py 
new_mesh = create_periodic_mesh(mesh, indicator, mapping)

#exit()
new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)

# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "periodic_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(new_mesh)

exit()


V = dolfinx.fem.functionspace(new_mesh, ("Lagrange", 2, (new_mesh.geometry.dim, )))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + 0.1*ufl.inner(u, v) * ufl.dx
x = ufl.SpatialCoordinate(new_mesh)
f = ufl.as_vector([10*x[0], 10**x[0]*ufl.sin(0.5*np.pi * x[1])])
L = ufl.inner(f, v) * ufl.dx
import dolfinx.fem.petsc
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[], petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
uh = problem.solve()


V_out = dolfinx.fem.functionspace(new_mesh, ("DG", 2, (new_mesh.geometry.dim, )))
u_out = dolfinx.fem.Function(V_out)
u_out.interpolate(uh)
with dolfinx.io.VTXWriter(new_mesh.comm, "u_periodic.bp", [u_out]) as writer:
    writer.write(0.0)