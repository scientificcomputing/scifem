# Create a periodic mesh in parallel
# SPDX-License-Identifier: MIT
# Author: Jørgen S. Dokken

from mpi4py import MPI
import numpy as np
import dolfinx
import ufl
import numpy.typing as npt


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
    global_vertices = vertex_map.local_to_global(closest_vertex)

    
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


    # Compute reduced index map without indicator vertices
    num_vertices_local = mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts
    parent_to_sub = np.full(num_vertices_local, -1, dtype=np.int32)
    parent_to_sub[sub_to_parent] = np.arange(sub_to_parent.size, dtype=np.int32)


    # Check if index is already in (reduced) vertex map
    # If not, add it to a ghost list
    parent_global_vertices = vertex_map.local_to_global(sub_to_parent)
    replacement_map = np.arange(num_vertices_local, dtype=np.int32)
    # Replace original vertex by its reduced index (after removing indicator vertices)
    replacement_map[np.arange(sub_to_parent.size)] = sub_to_parent
    new_ghosts = []
    new_owners = []
    reduced_ghosts = sub_map_without_ghosts.ghosts
    reduced_ghost_owners = sub_map_without_ghosts.owners
    for grv, grvo, sv in zip(global_replacement_vertex, global_replacement_owner, indicator_vertices, strict=True):
        grv_pos = np.argwhere(parent_global_vertices == grv)
        if grv_pos.shape[0] == 0:
            replacement_map[sv] = sub_map_without_ghosts.size_local +  len(reduced_ghosts) + len(new_ghosts)
            new_ghosts.append(grv)
            new_owners.append(grvo)
        else:
            replacement_map[sv] = grv_pos[0,0]

    new_ghosts = np.hstack([reduced_ghosts, new_ghosts]).astype(np.int64)
    new_owners = np.hstack([reduced_ghost_owners, new_owners]).astype(np.int32)
    new_local_size = int(sub_map_without_ghosts.size_local)
    new_vertex_map = dolfinx.common.IndexMap(mesh.comm, new_local_size,  new_ghosts, new_owners)

    # Convert old vertex_to_dofmap to reduced set
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    new_c = replacement_map[c_to_v.array]
    new_o = c_to_v.offsets.copy()
    new_c_to_v = dolfinx.graph.adjacencylist(new_c, new_o)
    new_v_to_v = dolfinx.graph.adjacencylist(np.arange(len(sub_to_parent), dtype=np.int32))


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



mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10,10)



def indicator(x):
    return np.isclose(x[0], 0.0)

def mapping(x):
    values = x.copy()
    values[0] += 1
    return values


new_mesh = create_periodic_mesh(mesh, indicator, mapping)
new_mesh.topology.create_connectivity(new_mesh.topology.dim, new_mesh.topology.dim-1)

exit()
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "periodic_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(new_mesh)

# exit()


V = dolfinx.fem.functionspace(new_mesh, ("N1curl", 2, (new_mesh.geometry.dim, )))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx
x = ufl.SpatialCoordinate(new_mesh)
f = ufl.as_vector([5*x[0]*ufl.sin(3*np.pi * x[0])+2*x[1], x[0]*ufl.sin(3*np.pi * x[1])])
L = ufl.inner(f, v) * ufl.dx
import dolfinx.fem.petsc
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[], petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
uh = problem.solve()


V_out = dolfinx.fem.functionspace(new_mesh, ("DG", 2, (new_mesh.geometry.dim, )))
u_out = dolfinx.fem.Function(V_out)
u_out.interpolate(uh)
with dolfinx.io.VTXWriter(new_mesh.comm, "u_periodic.bp", [u_out]) as writer:
    writer.write(0.0)