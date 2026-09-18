# Read the periodic node correspondence gmsh stores in a model
# SPDX-License-Identifier: MIT

"""Turn a gmsh model's ``$Periodic`` section into a vertex correspondence.

gmsh already knows which nodes a periodic boundary identifies -- it built the mesh that
way -- so the pairing need not be recovered by mapping coordinates and searching for the
nearest vertex. That also makes rotational and reflective periodicity work without the
caller writing the transform by hand.

What gmsh stores is a *relation*, not a function. Pairs are recorded per model entity,
including the dimension-0 point entities, so a corner node appears several times with
different masters: on a doubly periodic square the node at (1,1) is paired with (0,1)
through the right-hand curve and with (1,0) through the top curve. Both routes lead to
(0,0), and resolving to that common root is what this module does.
"""

from __future__ import annotations


import dolfinx
import numpy as np
from .mesh import create_periodic_mesh_from_igi, DEFAULT_TAG_BASE
from .utils import PeriodicNodes, resolve_to_roots


def extract_gmsh_periodic_nodes(
    model, include_high_order: bool = False, tol: float = 1e-8
) -> PeriodicNodes:
    """Collect the ``$Periodic`` node pairs of `model`, resolved to roots.

    Runs where the gmsh model lives, so serially on the reading rank.

    Args:
        model: An initialised ``gmsh.model`` carrying a meshed, periodic geometry.
        include_high_order: Keep the nodes that are not cell vertices. The correspondence
            this feeds is between *vertices*, and on a higher-order mesh
            ``entities_to_geometry(mesh, 0, vertices)`` returns each vertex's corner node,
            so the default is what is wanted. The resolution below treats the extra nodes
            no differently, so the flag costs nothing either way.
        tol: Absolute tolerance for checking each pair against the affine transform gmsh
            recorded with it. Pairs whose entity stored no transform are not checked.

    Returns:
        The pairs, as :class:`PeriodicNodes`.

    Raises:
        RuntimeError: If the pairs cycle, disagree on a root, or contradict the affine
            transform recorded with them.
    """
    slaves, masters, hops = [], [], []
    for dim, tag in model.getEntities():
        master_tag, node_tags, master_node_tags, affine = model.mesh.getPeriodicNodes(
            dim, tag, include_high_order
        )
        # gmsh returns the entity itself as its own master when it is not periodic.
        if master_tag == tag or len(node_tags) == 0:
            continue
        s = np.asarray(node_tags, dtype=np.int64) - 1
        m = np.asarray(master_node_tags, dtype=np.int64) - 1
        slaves.append(s)
        masters.append(m)
        hops.append((s, m, np.asarray(affine, dtype=np.float64)))

    all_node_tags, all_coords, _ = model.mesh.getNodes()
    num_nodes_global = int(np.asarray(all_node_tags, dtype=np.int64).max())

    if not slaves:
        return PeriodicNodes(num_nodes_global=num_nodes_global)

    slave = np.concatenate(slaves)
    master = np.concatenate(masters)

    # Coordinates by 0-based tag, for the affine check.
    coords = np.zeros((num_nodes_global, 3), dtype=np.float64)
    coords[np.asarray(all_node_tags, dtype=np.int64) - 1] = np.asarray(
        all_coords, dtype=np.float64
    ).reshape(-1, 3)
    for s, m, affine in hops:
        # gmsh stores a 4x4 row-major matrix, or nothing at all for some entities.
        if affine.size != 16:
            continue
        matrix = affine.reshape(4, 4)
        mapped = coords[m] @ matrix[:3, :3].T + matrix[:3, 3]
        gap = np.linalg.norm(mapped - coords[s], axis=1)
        if (gap > tol).any():
            i = int(np.argmax(gap))
            raise RuntimeError(
                "A `$Periodic` pair does not satisfy the affine transform recorded with"
                f" it: node tags (1-based) {int(m[i]) + 1} and {int(s[i]) + 1} are"
                f" {gap[i]:.3e} apart after the transform, tolerance {tol:.3e}."
            )

    unique_slave, root = resolve_to_roots(slave, master)
    assert not np.isin(root, unique_slave).any(), "a root is itself replaced"
    return PeriodicNodes(unique_slave, root, num_nodes_global)


def read_periodic_mesh_from_msh(
    filename,
    comm,
    rank: int = 0,
    gdim: int = 3,
    partitioner=None,
    tag_base: int = DEFAULT_TAG_BASE,
    **kwargs,
):
    """Read a ``.msh`` file and make the mesh periodic from its ``$Periodic`` section.

    Owns the gmsh session, because the pairs have to be read out of the model *before* it
    is finalized and the usual readers finalize it on the way out.

    Collective.

    Args:
        filename: The ``.msh`` file. Read on `rank` only.
        comm: The communicator to distribute the mesh over.
        rank: The rank that reads the file.
        gdim: Geometric dimension of the mesh.
        partitioner: Cell partitioner, passed through to ``model_to_mesh``.
        tag_base: Passed through to :func:`script.create_periodic_mesh_from_gmsh`.
        kwargs: Further arguments for ``model_to_mesh``, such as ``ghost_mode`` where the
            installed DOLFINx takes it there.

    Returns:
        ``(periodic_mesh, replaced_vertices, replacement_map)``, as
        :func:`script.create_periodic_mesh`.
    """
    import gmsh

    started_here = False
    try:
        if comm.rank == rank:
            if not gmsh.isInitialized():
                gmsh.initialize()
                started_here = True
            gmsh.model.add("periodic mesh from file")
            gmsh.merge(str(filename))
            pairs = extract_gmsh_periodic_nodes(gmsh.model)
        else:
            pairs = PeriodicNodes(0)

        mesh_data = dolfinx.io.gmsh.model_to_mesh(
            gmsh.model, comm, rank, gdim=gdim, partitioner=partitioner, **kwargs
        )
    finally:
        if started_here and gmsh.isInitialized():
            gmsh.finalize()

    mesh = getattr(mesh_data, "mesh", mesh_data)
    return create_periodic_mesh_from_igi(
        mesh,
        pairs.slave,
        pairs.master,
        pairs.num_nodes_global,
        root=rank,
        tag_base=tag_base,
    )
