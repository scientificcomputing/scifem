import dolfinx
import basix.ufl
from pathlib import Path
import numpy as np
import numpy.typing as npt
import warnings


def apply_mri_transform(
    path: Path,
    coordinates: npt.NDArray[np.floating],
    vox2ras_transform: npt.NDArray[np.floating] | None = None,
    use_tkr: bool = False,
) -> npt.NDArray[np.inexact]:
    """Given a path to a set of MRI voxel data, return the data evaluated at the given coordinates.

    Args:
        path: Path to the MRI data, should be a file format supported by nibabel.
        coordinates: Coordinates to evaluate the MRI data at.
        vox2ras_transform: Optional transformation matrix to convert from voxel to ras coordinates.
            If None, use the transformation matrix from the given MRI data.
        use_tkr: If true, use the old freesurfer `tkregister`, see:
            https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg69541.html
            for more details. Else use the standard VOX2RAS transform (equivalent to attaching
            an affine map to a nibabel image.)

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
    # Check if the image is in the correct orientation
    orientation = nibabel.aff2axcodes(image.affine)
    if orientation != ("R", "A", "S"):
        warnings.warn(f"Image orientation is {orientation}, expected (R, A, S). ")

    # Define mgh_header depending on MRI data types (i.e. nifti or mgz)
    if isinstance(image, nibabel.freesurfer.mghformat.MGHImage):
        mgh_header = image.header
    elif isinstance(image, nibabel.nifti1.Nifti1Image):
        mgh = nibabel.MGHImage(image.dataobj, image.affine)
        mgh_header = mgh.header
    else:
        raise ValueError(
            f"Unsupported image type: {type(image)} - Supported types are mgz and nifti"
        )

    # This depends on how one uses FreeSurfer
    if vox2ras_transform is not None:
        vox2ras = vox2ras_transform
    else:
        if use_tkr:
            vox2ras = mgh_header.get_vox2ras_tkr()
        else:
            # VOX to ras explanation: https://surfer.nmr.mgh.harvard.edu/ftp/articles/vox2ras.pdf
            vox2ras = mgh_header.get_vox2ras()
    # Check shape
    if vox2ras.shape != (4, 4):
        raise ValueError(f"vox2ras transform must be a 4x4 matrix, got shape {vox2ras.shape}")

    ras2vox = np.linalg.inv(vox2ras)

    ijk_vectorized = naff.apply_affine(ras2vox, coordinates)
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
    vox2ras_transform: npt.NDArray[np.floating] | None = None,
    use_tkr: bool = False,
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
    data = apply_mri_transform(Path(mri_data_path), midpoints, vox2ras_transform, use_tkr=use_tkr)
    et = dolfinx.mesh.meshtags(mesh, edim, entities, data.astype(np.int32))
    return et


def read_mri_data_to_function(
    mri_data_path: str | Path,
    mesh: dolfinx.mesh.Mesh,
    cells: npt.NDArray[np.int32] | None = None,
    vox2ras_transform: npt.NDArray[np.floating] | None = None,
    degree: int = 0,
    dtype: np.inexact = dolfinx.default_scalar_type,
    use_tkr: bool = False,
) -> dolfinx.fem.Function:
    """Read in MRI data over a set of cells in the mesh and attach it to an appropriate function.

    Args:
        mri_data_path: Path to MRI data
        mesh: The mesh to attach the MRI data to.
        cells: Subset of cells to evaluate the MRI data at. If None, all cells are used.
        vox2ras_transform: Optional transformation matrix to convert from voxel to ras coordinates.
            If None, use the transformation matrix from the given MRI data.
        degree: Degree of the (quadrature) function space to attach the MRI data to. Defaults to 0.
            If degree is 0, use a DG-0 space instead of a quadrature space, to simplify
            post-processing.
        dtype: Data type used for input data. Can be used for rounding data.
        use_tkr: If true, use the old freesurfer `tkregister`, see:
            https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg69541.html
            for more details. Else use the standard VOX2RAS transform (equivalent to attaching
            an affine map to a nibabel image.)


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
    data = apply_mri_transform(
        Path(mri_data_path), dof_positions[dofmap_pos], vox2ras_transform, use_tkr=use_tkr
    )

    v = dolfinx.fem.Function(Vh)
    v.x.array[dofmap_pos] = data.astype(dtype)
    return v
