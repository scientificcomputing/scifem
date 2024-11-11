# TODO

1.  Create interface that reads in the original global index from MSH directly
```python
from mpi4py import MPI
import dolfinx.io
import numpy.typing as npt
import gmsh

gmsh.initialize()
gmsh.model.add("Mesh from file")
gmsh.merge("mesh.msh")
phys_grps = gmsh.model.getPhysicalGroups()
periodic_entities: dict[
    tuple[int, int], tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]
] = {}

for dim, tag in phys_grps:
    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
    node_tags = []
    master_tags = []
    for entity in entities:
        tag_master, nodetags, nodetagsmasster, affinetransform = (
            gmsh.model.mesh.getPeriodicNodes(dim, entity)
        )
        print(nodetags, nodetagsmasster)
        node_tags.append(nodetags)
        master_tags.append(nodetagsmasster)
    s_tags = np.hstack(node_tags)
    m_tags = np.hstack(master_tags)
    assert len(s_tags) == len(m_tags)
    assert len(s_tags) == len(np.unique(s_tags))
    assert len(m_tags) == len(np.unique(m_tags))
    if len(s_tags) > 0:
        periodic_entities[(dim, tag)] = (s_tags, m_tags)
```
This matches the input global indices of the mesh geometry, and can be used for determining communcation pattern through a third party communicator.

1. Create sub map based on the s_tags being removed from each process.
2. For the process in possession of an `s_tags[i]`, check if:
   a. m_tags[i] in already on the process, if so: update the `replacement_map` with this index.
   b. Prepare ghost cells for all those connected to the vertex (cell_indices, original_cell_index, topology dm, topology owners).
   c. Send this data to process determined by https://github.com/jorgensd/adios4dolfinx/blob/main/src/adios4dolfinx/utils.py#L116-L135 for each vertex.
        As every process knowns where data is coming from, as all read the igi one should be able to determine the order of the incoming data.
        Should probably first send a message setting up communication directly between the processes, i.e. for process having s_tags[i], send index of current rank to process_owner.
        Similarly for `m_tags[i]` send ranks that have this index.
3. For the process that has a `m_tags[i]`, prepare ghost cells in the same way as 2. 
