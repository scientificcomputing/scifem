from .utils import VertexCorrespondence
from mpi4py import MPI
from ..mpi_utils import broadcast_marked_entities
import dolfinx
import numpy as np


def match_vertices_geometric(
    mesh, indicator, mapping_function, max_chain_length: int | None = None
) -> VertexCorrespondence:
    """Pair up the vertices of the seam by evaluating `mapping_function` on them.

    Selects the seam with `indicator`, moves each selected vertex with `mapping_function`,
    and snaps the image onto the nearest vertex of the mesh, checking that it actually
    landed there.

    Args:
        mesh: The mesh to make periodic.
        indicator: Marks the vertices to be replaced, given coordinates as ``(3, n)``.
        mapping_function: Maps a marked vertex to the one it is identified with, given
            coordinates as ``(3, n)``.
        max_chain_length: How many times `mapping_function` may be re-applied to reach a
            vertex outside `indicator`, for a mapping that applies one offset per call and
            so needs several passes to carry a corner to its root. Defaults to
            ``mesh.topology.dim``, the number of directions such a mesh can be periodic
            in. Exceeding it raises, which is how a cyclic mapping is caught.

    Returns:
        The correspondence :py:mod:`scifem.periodic` rebuilds from.
    """
    comm = mesh.comm
    if max_chain_length is None:
        max_chain_length = mesh.topology.dim

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # Find the vertices of the seam through the indicator function, and the facets given
    # up with them. Both are broadened so that the two sides agree on what is replaced.
    indicator_vertices = broadcast_marked_entities(
        mesh, 0, dolfinx.mesh.locate_entities_boundary(mesh, 0, indicator)
    )
    indicator_facets = broadcast_marked_entities(
        mesh,
        mesh.topology.dim - 1,
        dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, indicator),
    )

    geom_index = dolfinx.mesh.entities_to_geometry(mesh, 0, indicator_vertices).reshape(-1)
    owned_vertex_coords = mesh.geometry.x[geom_index]

    # A geometric tolerance has to be a length. `np.finfo(...).eps` describes relative
    # precision near 1.0, so as an absolute padding it stops meaning anything once the mesh
    # sits away from the origin: at coordinates around 1e6 a single representable double is
    # already 1.2e-10, larger than 10000 * eps. Take a small fraction of the smallest cell
    # instead, floored by the rounding error of the coordinates themselves, which grows with
    # distance from the origin.
    _cell_map = mesh.topology.index_map(mesh.topology.dim)
    _cell_sizes = dolfinx.cpp.mesh.h(
        mesh._cpp_object,
        mesh.topology.dim,
        np.arange(_cell_map.size_local + _cell_map.num_ghosts, dtype=np.int32),
    )
    h_min = comm.allreduce(_cell_sizes.min() if _cell_sizes.size else np.inf, op=MPI.MIN)
    coord_scale = comm.allreduce(
        np.abs(mesh.geometry.x).max() if mesh.geometry.x.size else 0.0, op=MPI.MAX
    )
    eps = max(
        1e-6 * h_min,
        100 * np.finfo(mesh.geometry.x.dtype).eps * max(1.0, coord_scale),
    )

    # Map vertices to new coordinates
    mapped_vertex_coords = np.ascontiguousarray(mapping_function(owned_vertex_coords.T).T)

    # Follow the mapping to a vertex outside `indicator`; see `max_chain_length` above.
    # Entirely local: `indicator` and `mapping_function` are pointwise in the coordinates,
    # so a process may run out of chains to follow before another does.
    local_unresolved = False
    for _ in range(max_chain_length):
        still_indicated = np.asarray(indicator(mapped_vertex_coords.T), dtype=np.bool_)
        if not still_indicated.any():
            break
        mapped_vertex_coords[still_indicated] = mapping_function(
            mapped_vertex_coords[still_indicated].T
        ).T
    else:
        local_unresolved = True

    # Collective check to check that all vertex mappings have been resolved.
    if comm.allreduce(int(local_unresolved), op=MPI.SUM) > 0:
        raise RuntimeError(
            f"`mapping_function` did not reach a vertex outside `indicator` within"
            f" {max_chain_length} applications. Either the two functions disagree, or the"
            " mapping cycles: an indicator vertex is mapped onto another that maps back."
        )

    # For each vertex that will be replaced, find which process should take it over
    vertex_ownership_data = dolfinx.geometry.determine_point_ownership(
        mesh, mapped_vertex_coords, padding=eps
    )
    # On process that has taken over a vertex, find the closest vertex (local to proc) that
    # will be its replacement
    acquired_vertex_coords = vertex_ownership_data.dest_points
    potential_closest_vertices = dolfinx.mesh.compute_incident_entities(
        mesh.topology, vertex_ownership_data.dest_cells, mesh.topology.dim, 0
    )
    closest_vertex_bb_tree = dolfinx.geometry.bb_tree(
        mesh, 0, entities=potential_closest_vertices, padding=eps
    )
    closest_vertex_mid_tree = dolfinx.geometry.create_midpoint_tree(
        mesh, 0, potential_closest_vertices
    )
    closest_vertex = dolfinx.geometry.compute_closest_entity(
        closest_vertex_bb_tree,
        closest_vertex_mid_tree,
        mesh,
        acquired_vertex_coords,
    )

    # `compute_closest_entity` returns the closest candidate whether or not it is anywhere
    # near the query point, so check that the mapped point really landed on it. A mapping
    # wrong by a whole cell otherwise gives a valid-looking mesh that is not periodic.
    closest_vertex_coords = mesh.geometry.x[
        dolfinx.mesh.entities_to_geometry(mesh, 0, closest_vertex).reshape(-1)
    ]
    snap_distance = np.linalg.norm(closest_vertex_coords - acquired_vertex_coords, axis=1)
    num_unsnapped = int(np.count_nonzero(snap_distance > eps))
    # Collective: the condition is reduced so that either every process raises or none
    # does. A one-sided raise would leave the others blocked in the rebuild's exchanges.
    total_unsnapped = comm.allreduce(num_unsnapped, op=MPI.SUM)
    if total_unsnapped > 0:
        worst = comm.allreduce(float(np.max(snap_distance, initial=0.0)), op=MPI.MAX)
        raise RuntimeError(
            f"`mapping_function` did not map {total_unsnapped} vertices onto a vertex of"
            f" the mesh; the largest gap between a mapped point and the closest vertex is"
            f" {worst:.3e}. Every mapped point has to land on the vertex it is meant to be"
            " identified with, otherwise the result is a mesh with the expected vertex and"
            " cell counts that is not periodic."
        )

    return VertexCorrespondence(
        indicator_vertices=indicator_vertices,
        indicator_facets=indicator_facets,
        src_owner=vertex_ownership_data.src_owner,
        dest_owner=vertex_ownership_data.dest_owner,
        partner_vertex=closest_vertex,
    )
