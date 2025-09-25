from mpi4py import MPI
import dolfinx
import basix
import numpy as np
from . import _scifem  # type: ignore
from collections.abc import Sequence
from packaging.version import Version


def create_real_functionspace(
    mesh: dolfinx.mesh.Mesh, value_shape: tuple[int, ...] = ()
) -> dolfinx.fem.FunctionSpace:
    """Create a real function space.

    Args:
        mesh: The mesh the real space is defined on.
        value_shape: The shape of the values in the real space.

    Returns:
        The real valued function space.
    Note:
        For scalar elements value shape is ``()``.

    """

    dtype = mesh.geometry.x.dtype
    ufl_e = basix.ufl.element(
        "P", mesh.basix_cell(), 0, dtype=dtype, discontinuous=True, shape=value_shape
    )

    if (dtype := mesh.geometry.x.dtype) == np.float64:
        cppV = _scifem.create_real_functionspace_float64(mesh._cpp_object, value_shape)
    elif dtype == np.float32:
        cppV = _scifem.create_real_functionspace_float32(mesh._cpp_object, value_shape)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dolfinx.fem.FunctionSpace(mesh, ufl_e, cppV)


def create_space_of_simple_functions(
    mesh: dolfinx.mesh.Mesh,
    cell_tag: dolfinx.mesh.MeshTags,
    tags: Sequence[int | np.int32 | np.int64],
    value_shape: tuple[()] | tuple[int | tuple[int]] | None = None,
) -> dolfinx.fem.FunctionSpace:
    """Create a space of simple functions.

    This is a space that represents piecewise constant functions of `N` patches,
    where `N` is the number of input `tags`.
    Each patch is defined by the cells in `cell_tag` which is marked with the
    corresponding tag value.

    Note:
        All cells are expected to have a tag in `tags`.

    Args:
        mesh: The mesh the function space is defined on.
        cell_tag: The mesh tags defining the different patches of cells.
        tags: The set of unique values within `cell_tag` that defines the different patches.
        value_shape: The shape of the values in the space.

    Returns:
        The space of simple functions.

    """
    value_shape = () if value_shape is None else value_shape
    if mesh.topology._cpp_object != cell_tag.topology:
        raise ValueError("Topology of cell tag is not the mesh topology")
    if cell_tag.dim != mesh.topology.dim:
        raise ValueError(
            f"The dimension of the input meshtag is {cell_tag.dim}, expected {mesh.topology.dim}"
        )

    cell_map = mesh.topology.index_map(mesh.topology.dim)

    # Sanity check cell tags
    if cell_map.size_local + cell_map.num_ghosts != len(cell_tag.indices):
        raise ValueError("Every cell in the cell tag must have a value assigned to it.")
    if not np.isin(cell_tag.values, tags).all():
        raise ValueError("All values in the cell tag must be in the input tags.")

    # Determine the owner of the material degrees of freedom.
    # It will be the process that owns the cell with global index 0
    is_owner = cell_map.local_range[0] == 0 and cell_map.size_local > 0

    # Determine number of dofs and ghosts on each process
    el = basix.ufl.element("P", mesh.basix_cell(), 0, shape=(), discontinuous=True)
    num_dofs = len(tags) * is_owner
    num_ghosts = len(tags) * (not is_owner)
    (_, ghost_owner_rank) = mesh.comm.allreduce((is_owner, mesh.comm.rank), op=MPI.MAXLOC)

    # Create index map describing the distribution of dofs across processes
    ghosts = np.arange(num_ghosts, dtype=np.int64)
    owners = np.full(num_ghosts, ghost_owner_rank, dtype=np.int32)
    if Version(dolfinx.__version__) > Version("0.9.0"):
        imap_kwargs = {"tag": 321}
    else:
        imap_kwargs = {}
    dof_imap = dolfinx.common.IndexMap(mesh.comm, num_dofs, ghosts, owners, **imap_kwargs)

    # Create element dof layout (1 dof per cell, based of the DG-0 element)
    value_size = int(np.prod(value_shape))
    index_map_bs = value_size
    bs = value_size

    e_layout = dolfinx.cpp.fem.ElementDofLayout(
        value_size, el.entity_dofs, el.entity_closure_dofs, [], []
    )

    # Create dofmap mapping local dofs to position in `tags` input.
    adj_flattened = np.zeros_like(cell_tag.values)
    for i, tag in enumerate(tags):
        adj_flattened[cell_tag.find(tag)] = i

    # Create dofmap
    dofmap_adj = dolfinx.cpp.graph.AdjacencyList_int32(
        adj_flattened, np.arange(len(cell_tag.values) + 1, dtype=np.int32)
    )
    cpp_dofmap = dolfinx.cpp.fem.DofMap(e_layout, dof_imap, index_map_bs, dofmap_adj, bs)

    # Create function space
    try:
        cpp_el = dolfinx.cpp.fem.FiniteElement_float64(
            el.basix_element._e, block_shape=value_shape, symmetric=False
        )
        cpp_space = dolfinx.cpp.fem.FunctionSpace_float64(mesh._cpp_object, cpp_el, cpp_dofmap)
    except TypeError:
        cpp_el = dolfinx.cpp.fem.FiniteElement_float64(
            el.basix_element._e, block_size=bs, symmetric=False
        )
        cpp_space = dolfinx.cpp.fem.FunctionSpace_float64(
            mesh._cpp_object, cpp_el, cpp_dofmap, value_shape=value_shape
        )

    _el = basix.ufl.element("P", mesh.basix_cell(), 0, shape=value_shape, discontinuous=True)
    V = dolfinx.fem.FunctionSpace(mesh, _el, cpp_space)
    return V
