import dolfinx
import basix.ufl
from pathlib import Path
import numpy as np
import numpy.typing as npt


def apply_mri_transform(
    path: Path, coordinates: npt.NDArray[np.floating], use_tkr: bool = False
) -> npt.NDArray[np.inexact]:
    """Given a path to a set of MRI voxel data, return the data evaluated at the given coordinates.

    Args:
        path: Path to the MRI data, should be a file format supported by nibabel.
        coordinates: Coordinates to evaluate the MRI data at.
        use_tkr: If true, use the old freesurfer `tkregister`, see:
            https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg69541.html
            for more details. Else use the standard VOX2RAS transform (equivalent to attaching
            an affine map to a nibabel image.

    Returns:
        The data evaluated at these points.
    """
    try:
        import nibabel
        import nibabel.affines as naff
    except ImportError:
        raise ImportError("This function requires the nibabel package to be installed")
    image = nibabel.load(path)
    data = image.get_fdata()
    # This depends on how one uses FreeSurfer
    if use_tkr:
        vox2ras = image.header.get_vox2ras_tkr()
    else:
        # VOX to ras explanation: https://surfer.nmr.mgh.harvard.edu/ftp/articles/vox2ras.pdf
        vox2ras = image.header.get_vox2ras()
    ras2vox = np.linalg.inv(vox2ras)
    ijk_vectorized = naff.apply_affine(ras2vox, coordinates)

    # Round indices to nearest integer
    # The file standard assumes that the voxel coordinates refer to the center of each voxel.
    # https://brainder.org/2012/09/23/the-nifti-file-format/
    ijk_rounded = np.rint(ijk_vectorized).astype("int")

    assert np.all(ijk_rounded >= 0)
    return data[ijk_rounded[:, 0], ijk_rounded[:, 1], ijk_rounded[:, 2]]


def read_mri_data_to_tag(
    mri_data_path: str | Path,
    mesh: dolfinx.mesh.Mesh,
    edim: int,
    entities: npt.NDArray[np.int32] | None = None,
) -> dolfinx.mesh.MeshTags:
    """Read in MRI data over a set of entities in the mesh and attach it to a mesh tag.

    Args:
        mri_data_path: Path to MRI data
        mesh: The mesh to attach the MRI data to.
        edim: Topological dimension of entities to attach the MRI data to.
        entities: List of entities (local to process) to evaluate the MRI data at. If None, all
            local entities are used.

    Returns:
        The mesh tag with the attached MRI data.
    """
    if entities is None:
        mesh.topology.create_entities(edim)
        entity_map = mesh.topology.index_map(edim)
        num_entities = entity_map.size_local + entity_map.num_ghosts
        entities = np.arange(num_entities, dtype=np.int32)

    midpoints = dolfinx.mesh.compute_midpoints(mesh, edim, entities)
    data = apply_mri_transform(Path(mri_data_path), midpoints)
    et = dolfinx.mesh.meshtags(mesh, edim, entities, data.astype(np.int32))
    return et


def read_mri_data_to_function(
    mri_data_path: str | Path,
    mesh: dolfinx.mesh.Mesh,
    cells: npt.NDArray[np.int32] | None = None,
    degree: int = 0,
    dtype: np.inexact = dolfinx.default_scalar_type,
) -> dolfinx.fem.Function:
    """Read in MRI data over a set of cells in the mesh and attach it to an appropriate function.

    Args:
        mri_data_path: Path to MRI data
        mesh: The mesh to attach the MRI data to.
        cells: Subset of cells to evaluate the MRI data at. If None, all cells are used.
        degree: Degree of the (quadrature) function space to attach the MRI data to. Defaults to 0.
            If degree is 0, use a DG-0 space instead of a quadrature space, to simplify
            post-processing.
        dtype: Data type used for input data. Can be used for rounding data.

    Raises:
        ValueError: If degree is negative.

    Returns:
        The function with the attached MRI data.
    """
    if not degree >= 0:
        raise ValueError("Degree must be a non-negative integer")

    if degree == 0:
        el = basix.ufl.element("Lagrange", mesh.basix_cell(), 0, discontinuous=True)
    else:
        el = basix.ufl.quadrature_element(scheme="default", degree=degree, cell=mesh.basix_cell())

    if cells is None:
        cell_map = mesh.topology.index_map(mesh.topology.dim)
        num_cells = cell_map.size_local + cell_map.num_ghosts
        cells = np.arange(num_cells, dtype=np.int32)

    Vh = dolfinx.fem.functionspace(mesh, el)
    dof_positions = Vh.tabulate_dof_coordinates()
    dofmap = Vh.dofmap.list
    dofmap_pos = dofmap[cells].flatten()
    data = apply_mri_transform(Path(mri_data_path), dof_positions[dofmap_pos])

    v = dolfinx.fem.Function(Vh)
    v.x.array[dofmap_pos] = data.astype(dtype)
    return v
