""" Fixing overconstrained cells in a mesh

Copyright (C) 2025   Jørgen Dokken (dokken@simula.no)
Copyright (C) 2025   Cécile Daversin-Catty (cecile@simula.no)
"""

import warnings

import gc
import numpy as np

from ufl import Mesh as ufl_mesh

from dolfinx.io import (
    distribute_entity_data
)
from dolfinx.graph import (
    adjacencylist
)
from dolfinx.mesh import (
    MeshTags,
    compute_incident_entities,
    compute_midpoints,
    entities_to_geometry,
    create_mesh,
    meshtags_from_entities
)

__all__ = [
    "fix_overconstrained_cells",
]


# Fix overconstrained cells in a mesh
# Greatly inspired from Jørgen's implementation https://gist.github.com/jorgensd/0b65aac1117a502cfc12f58b3d3c9508
class OverconstrainedMeshFix:
    """
    Class to fix overconstrained cells in a mesh.
    Overconstrained cells are those that have all their vertices on the boundary.
    This is necessary to avoid issues with the mesh generation.
    """

    def __init__(self, mesh, cell_function, facet_function, cell_tags=None):
        """
        Initialize the OverconstrainedMeshFix class.
        Args:
            mesh: The mesh to be repaired.
            cell_function: The cell function to be transferred.
            facet_function: The facet function to be transferred.
            cell_tags(Optional): meshtags marking subdomains (subsets of cells) for which boundary needs to be repaired.
                                 The subdomains included in the repaired are those with tags > 0.
        """
        self.mesh = mesh
        assert mesh.comm.size == 1, "This function is not working in parallel"
        print(f"Mesh has {mesh.topology.index_map(mesh.topology.dim).size_local} cells and {mesh.topology.index_map(0).size_local} points")

        self.cell_function = cell_function
        assert isinstance(self.cell_function, MeshTags), "cell_function must be a MeshTags object"
        self.facet_function = facet_function
        assert isinstance(self.facet_function, MeshTags), "facet_function must be a MeshTags object"
        self.cell_tags = cell_tags
        if cell_tags is not None:
            assert isinstance(self.cell_tags, MeshTags), "cell_tags must be a MeshTags object"

        self.initialize_mesh_characteristics()
        self.removed_cells = []  # Cells that are removed
        self.removed_facets = []  # Facets that are removed

    def initialize_mesh_characteristics(self):
        """Initialize mesh characteristics."""
        self.tdim = self.mesh.topology.dim
        assert self.tdim == 3, "This function only works for 3D meshes"

        self.num_vertices = self.mesh.topology.index_map(0).size_local
        self.mesh.topology.create_entities(self.tdim - 1)
        self.num_facets = self.mesh.topology.index_map(self.tdim - 1).size_local
        self.num_cells = self.mesh.topology.index_map(self.tdim).size_local

        # Recreate connectivity for cells, facets, and vertices
        self.mesh.topology.create_connectivity(self.tdim, self.tdim - 1)
        self.c_to_f = self.mesh.topology.connectivity(self.tdim, self.tdim - 1)
        self.mesh.topology.create_connectivity(self.tdim - 1, self.tdim)
        self.f_to_c = self.mesh.topology.connectivity(self.tdim - 1, self.tdim)
        self.mesh.topology.create_connectivity(self.tdim - 1, 0)
        self.f_to_v = self.mesh.topology.connectivity(self.tdim - 1, 0)
        self.c_to_v = self.mesh.topology.connectivity(self.tdim, 0)

    def __del__(self):
        """Destructor to clean up resources."""
        if self.mesh is not None:
            del self.mesh
        if self.c_to_f is not None:
            del self.c_to_f
        if self.f_to_c is not None:
            del self.f_to_c
        if self.f_to_v is not None:
            del self.f_to_v
        if self.c_to_v is not None:
            del self.c_to_v
        if self.removed_cells is not None:
            del self.removed_cells
        if self.new_cells_as_array is not None:
            del self.new_cells_as_array
        gc.collect()

    def fixed_mesh(self):
        """Return the mesh."""
        return self.mesh

    def updated_cell_function(self):
        """Return the (updated) cell function."""
        return self.cell_function

    def updated_cell_tags(self):
        """Return the (updated) cell tags."""
        return self.cell_tags

    def updated_facet_function(self):
        """Return the (updated) facet function."""
        return self.facet_function

    def nb_removed_cells(self):
        """Return the number of removed cells."""
        if self.removed_cells is None:
            return 0
        return len(self.removed_cells)

    def nb_new_cells(self):
        """Return the number of new cells created."""
        if self.new_cells_as_array is None:
            return 0
        return len(self.new_cells_as_array)

    def create_new_cells(self, current_cell, needed_vertex, new_vertex):
        """
        Given a cell `[v0, v1, v2, v3]`, a required vertex `v2`, and a new vertex `v`,
        `[v1,v2,v3,v], [v0,v2,v3,v], [v0,v1,v2,v]`
        """

        include = np.argwhere(current_cell == needed_vertex)
        assert len(include) == 1
        # Creating 3 new cells with 4 vertices each
        new_cells = np.full((3, 4), -1, dtype=np.int64)
        mask = np.full(4, False, dtype=np.bool_)
        loop = np.arange(4)
        loop = np.delete(loop, include)
        for i, insert_pos in enumerate(loop):
            mask[:] = True
            mask[insert_pos] = False
            new_cells[i] = np.hstack([current_cell[mask], [new_vertex]])
        return new_cells

    def create_facet_marker_structure(self, facet, new_vertex):
        """Split facet of current cell in 3 and create marker structure
        Args:
            new_vertex: w
            facet: [v0,v2,v3]
        Returns [v0,v2, w], [v0,w,v3], [v2,v3,w]
        """
        mask = np.full(3, True, dtype=np.bool_)
        new_facets = np.full((3, 3), -1, dtype=np.int64)
        for i in range(3):
            mask[:] = True
            mask[i] = False
            new_facets[i] = np.hstack([facet[mask], [new_vertex]])
        return new_facets

    def is_overconstrained(self, cell_index: int) -> tuple[bool, np.int32 | None]:
        """Check if a boundary cell is overconstrained, i.e. all vertices are on the boundary.
        Args:
            cell_index: Index of the cell to check
        Returns:
            True if the cell is overconstrained, False otherwise
            index of the facet to be split if the cell is overconstrained, None otherwise
        """

        exterior_facets = []
        interior_facets = []
        interface_facets = []

        facets = self.c_to_f.links(cell_index)
        for facet in facets:
            if len(self.f_to_c.links(facet)) == 1:
                exterior_facets.append(facet)  # Exterior (global boundary)
            elif len(self.f_to_c.links(facet)) == 2:
                if self.cell_tags is not None:  # Interface (subdomain boundary)
                    other_cell_idx = np.argwhere(self.f_to_c.links(facet) != cell_index)[0]
                    other_cell = self.f_to_c.links(facet)[other_cell_idx]
                    if self.cell_tags.values[cell_index] > 0 and self.cell_tags.values[other_cell] == 0:
                        interface_facets.append(facet)
                    else:
                        interior_facets.append(facet)
                else:
                    interior_facets.append(facet)
            else:
                raise ValueError(f"Facet {facet} has {len(self.f_to_c.links(facet))} links, expected 1 or 2")

        exterior_facets = np.array(exterior_facets, dtype=np.int32)
        interior_facets = np.array(interior_facets, dtype=np.int32)
        interface_facets = np.array(interface_facets, dtype=np.int32)

        if not len(interior_facets) > 0:
            assert len(exterior_facets) == 4
            print(f"Cell {cell_index} with midpoint {compute_midpoints(self.mesh, self.tdim, np.array([cell_index], dtype=np.int32))} has no interior facets")
            self.removed_cells.append(cell_index)
            [self.removed_facets.append(f) for f in exterior_facets]
            return (True, None)

        exterior_vertices = compute_incident_entities(
            self.mesh.topology, exterior_facets, self.tdim - 1, 0
        )
        assert len(np.unique(exterior_vertices)) == len(exterior_vertices)

        interface_vertices = compute_incident_entities(
            self.mesh.topology, interface_facets, self.tdim - 1, 0
        )
        assert len(np.unique(interface_vertices)) == len(interface_vertices)

        # Splitting overconstrained cell at exterior boundary
        if len(exterior_vertices) == 4:
            print(f"Cell {cell_index} with midpoint {compute_midpoints(self.mesh, self.tdim, np.array([cell_index], dtype=np.int32))} has 4 exterior vertices")
            return (True, interior_facets[0])
        # Splitting overconstrained cell at boundary of tagged domain
        elif len(interface_vertices) == 4 and self.cell_tags.values[cell_index] > 0:
            print(f"Cell {cell_index} with midpoint {compute_midpoints(self.mesh, self.tdim, np.array([cell_index], dtype=np.int32))} has 4 interface vertices")
            return (True, interior_facets[0])
        else:
            # Cell is not overconstrained
            return (False, None)

    def fix_overconstrained_cells(self):

        # Number of entities
        num_vertices = self.mesh.topology.index_map(0).size_local
        num_cells = self.mesh.topology.index_map(self.tdim).size_local
        num_facets = self.mesh.topology.index_map(self.tdim - 1).size_local
        num_vertices = self.mesh.topology.index_map(0).size_local

        facet_midpoints = compute_midpoints(
            self.mesh, self.tdim - 1, np.arange(num_facets, dtype=np.int32)
        )

        # (Re-)initialize removed cells, facets, and new cells arrays
        self.removed_cells = []
        self.removed_facets = []
        self.new_cells_as_array = None

        # New mesh entities
        new_vertex_counter = 0
        new_vertex_coordinates = []
        new_cells = []
        # New mesh tags
        new_cell_marker_array = []
        new_cell_tags_array = []
        new_facet_marker_array = []
        new_marked_facets = []

        for i in range(num_cells):
            if i in self.removed_cells:
                # Cell has already been removed, skip it
                continue

            # Check if cell is overconstrained
            is_overconstrained, facet_idx = self.is_overconstrained(i)

            if is_overconstrained:
                if facet_idx is None:
                    # Cell is overconstrained and has already been removed
                    continue

                all_cells = self.f_to_c.links(facet_idx)
                other_cell = np.setdiff1d(all_cells, [i])[0]
                if other_cell in self.removed_cells:
                    warnings.warn("Cell already removed, should call this function again")
                    continue
                current_vertices = self.c_to_v.links(i)
                interior_facet_vertices = self.f_to_v.links(facet_idx)

                # Get position of new vertex on midpoint of facet
                coord = facet_midpoints[facet_idx]
                new_vertex_coordinates.append(coord)
                all_needs = np.setdiff1d(current_vertices, interior_facet_vertices)

                # Transfer facet marker if and only if it was in original tag
                pos = np.flatnonzero(self.facet_function.indices == facet_idx)
                if len(pos) > 0:
                    # Get all new sub-facets
                    new_marked_facets.append(
                        self.create_facet_marker_structure(
                            interior_facet_vertices, num_vertices + new_vertex_counter
                        )
                    )
                    for _ in range(3):
                        # Check if facet has been marked in previous grid
                        assert len(pos) == 1
                        new_facet_marker_array.append(self.facet_function.values[pos])
                    self.removed_facets.append(facet_idx)

                # Split troublesome cell in 3
                assert len(all_needs) == 1
                include = np.argwhere(current_vertices == all_needs[0])
                assert len(include) == 1
                new_cells.append(
                    self.create_new_cells(
                        current_vertices, all_needs[0], num_vertices + new_vertex_counter
                    )
                )

                other_cell_connectivity = self.c_to_v.links(other_cell)
                other_needs = np.setdiff1d(other_cell_connectivity, interior_facet_vertices)
                new_cells.append(
                    self.create_new_cells(
                        other_cell_connectivity,
                        other_needs[0],
                        num_vertices + new_vertex_counter,
                    )
                )

                for _ in range(3):
                    new_cell_marker_array.append(self.cell_function.values[i])
                for _ in range(3):
                    new_cell_marker_array.append(self.cell_function.values[other_cell])

                if self.cell_tags is not None:
                    # If cell tags are provided, transfer them to the new cells
                    for _ in range(3):
                        new_cell_tags_array.append(self.cell_tags.values[i])
                    for _ in range(3):
                        new_cell_tags_array.append(self.cell_tags.values[other_cell])

                new_vertex_counter += 1
                self.removed_cells.append(i)
                self.removed_cells.append(other_cell)
        new_marked_facets = np.array(new_marked_facets, dtype=np.int64).reshape(-1, 3)
        if len(new_facet_marker_array) > 0:
            new_facet_marker_array = np.hstack(new_facet_marker_array).astype(np.int32)
        print("All cells checked!")

        if len(new_cells) == 0:
            self.new_cells_as_array = np.zeros((0, 4), dtype=np.int64)
        else:
            self.new_cells_as_array = np.vstack(new_cells)
        assert len(self.new_cells_as_array) == len(new_cell_marker_array)

        if len(self.removed_cells) == 0:
            self.removed_cells = np.array([], dtype=np.int64)
        else:
            self.removed_cells = np.unique(np.hstack(self.removed_cells).astype(np.int64))

        # Gather all cells
        remaining_cells = np.arange(num_cells, dtype=np.int32)
        remaining_cells = np.delete(remaining_cells, self.removed_cells)

        all_cells = entities_to_geometry(
            self.mesh, self.tdim, remaining_cells
        )
        all_new_cells = np.vstack([all_cells, self.new_cells_as_array]).astype(np.int64)

        new_vertex_numbering = np.unique(all_new_cells.flatten())
        all_to_reduced_num_vertices = np.full(num_vertices + new_vertex_counter, -1, dtype=np.int64)
        all_to_reduced_num_vertices[new_vertex_numbering] = np.arange(len(new_vertex_numbering))
        all_new_cells = all_to_reduced_num_vertices[all_new_cells]

        # Gather all coordinates
        all_coords = np.zeros((num_vertices + new_vertex_counter, 3), dtype=np.float64)
        if new_vertex_counter > 0:
            all_new_vertex_coordinates = np.vstack(new_vertex_coordinates)
        else:
            all_new_vertex_coordinates = np.zeros((0, 3), dtype=np.float64)
        all_coords[:num_vertices, :] = self.mesh.geometry.x
        all_coords[num_vertices:, :] = all_new_vertex_coordinates
        all_coords = all_coords[new_vertex_numbering]

        ufl_domain = ufl_mesh(self.mesh._ufl_domain.ufl_coordinate_element())  # type: ignore
        new_mesh = create_mesh(
            self.mesh.comm, all_new_cells, all_coords, ufl_domain
        )
        new_mesh.topology.create_connectivity(new_mesh.topology.dim, 0)

        # Transfer markers
        print("Transferring cell markers")
        assert len(self.cell_function.indices) == num_cells
        mask = np.full(num_cells, True, dtype=np.bool_)
        mask[self.removed_cells] = False
        new_values = self.cell_function.values[mask]
        all_values = np.hstack([new_values, new_cell_marker_array])
        local_entities, local_values = distribute_entity_data(
            new_mesh, new_mesh.topology.dim, all_new_cells, all_values
        )
        adj = adjacencylist(local_entities)
        ct = meshtags_from_entities(
            new_mesh, new_mesh.topology.dim, adj, local_values.astype(np.int32, copy=False)
        )
        ct.name = self.cell_function.name

        ct_tags = None
        if self.cell_tags is not None:
            # Transfer cell tags if provided
            assert len(self.cell_tags.indices) == num_cells
            new_cell_tags_values = self.cell_tags.values[mask]
            all_cell_tags_values = np.hstack([new_cell_tags_values, new_cell_tags_array])
            local_tags_entities, local_tags_values = distribute_entity_data(
                new_mesh, new_mesh.topology.dim, all_new_cells, all_cell_tags_values
            )
            adj_tags = adjacencylist(local_tags_entities)
            ct_tags = meshtags_from_entities(
                new_mesh, new_mesh.topology.dim, adj_tags, local_tags_values.astype(np.int32, copy=False)
            )
            ct_tags.name = self.cell_tags.name

        # Create facet marker structure
        print("Transferring facet marker")
        facets_to_copy = self.facet_function.indices.copy()
        facets_to_keep = np.invert(np.isin(facets_to_copy, self.removed_facets))
        assert np.allclose(self.f_to_v.offsets[1:] - self.f_to_v.offsets[:-1], 3)
        conn_arr = self.f_to_v.array.reshape(-1, 3)
        new_facet_array = conn_arr[facets_to_copy[facets_to_keep], :]
        new_facet_values_array = self.facet_function.values[facets_to_keep]
        new_facet_values_array = np.hstack(new_facet_values_array, dtype=np.int32).reshape(-1)
        # Renumber the vertices of facets to align with the reduced set of vertices
        facet_connectivity = np.vstack([new_facet_array, new_marked_facets])
        facet_connectivity = all_to_reduced_num_vertices[facet_connectivity].astype(np.int64)
        assert (facet_connectivity != -1).all()
        facet_values = np.hstack([new_facet_values_array, new_facet_marker_array])

        local_entities, local_values = distribute_entity_data(
            new_mesh, new_mesh.topology.dim - 1, facet_connectivity, facet_values
        )
        new_mesh.topology.create_connectivity(new_mesh.topology.dim - 1, 0)
        adj = adjacencylist(local_entities)
        new_mesh.topology.create_connectivity(
            new_mesh.topology.dim - 1, new_mesh.topology.dim
        )
        ft = meshtags_from_entities(
            new_mesh,
            new_mesh.topology.dim - 1,
            adj,
            local_values.astype(np.int32, copy=False),
        )
        ft.name = self.facet_function.name

        # Update mesh and tags
        self.mesh = new_mesh
        self.cell_function = ct
        self.facet_function = ft
        self.cell_tags = ct_tags


def fix_overconstrained_cells(msh, label, bm, cell_tag=None, max_iter=10):
    """
    Fix overconstrained cells, i.e. cells having all their vertices on the boundary.
    This is necessary to avoid issues with the mesh generation.
    """
    fixing_overconstrained_cells = True
    iter = 0
    MeshFix = OverconstrainedMeshFix(msh, label, bm, cell_tags=cell_tag)

    while fixing_overconstrained_cells and iter < max_iter:
        print("Fixing overconstrained cells...")
        MeshFix.fix_overconstrained_cells()
        if MeshFix.nb_removed_cells() == 0 and MeshFix.nb_new_cells() == 0:
            fixing_overconstrained_cells = False
            print("No more overconstrained cells found.")
        else:
            print(f"{MeshFix.nb_removed_cells()} cells were removed - {MeshFix.nb_new_cells()} cells were added.")
        iter += 1
        MeshFix.__init__(MeshFix.fixed_mesh(),
                         MeshFix.updated_cell_function(),
                         MeshFix.updated_facet_function(),
                         cell_tags=MeshFix.updated_cell_tags())

    return MeshFix.fixed_mesh(), MeshFix.updated_cell_function(), MeshFix.updated_facet_function(), MeshFix.updated_cell_tags()
