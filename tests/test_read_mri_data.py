from mpi4py import MPI
import dolfinx
import numpy as np
from scifem.biomedical import read_mri_data_to_function, read_mri_data_to_tag
import pytest
import basix.ufl


@pytest.mark.parametrize("degree", [0, 1, 3])
@pytest.mark.parametrize("M", [25, 17])
@pytest.mark.parametrize("Nx", [7, 3])
@pytest.mark.parametrize("Ny", [23, 8])
@pytest.mark.parametrize("Nz", [9, 17])
@pytest.mark.parametrize("theta", [np.pi / 3, 0, -np.pi])
@pytest.mark.parametrize("translation", [np.array([0, 0, 0]), np.array([2.1, 1.3, 0.4])])
@pytest.mark.parametrize("mri_data_format", ["nifti", "mgh"])
@pytest.mark.parametrize("external_affine", [True, False])
@pytest.mark.parametrize("use_tkr", [False])
def test_read_mri_data_to_function(
    degree, M, Nx, Ny, Nz, theta, translation, tmp_path, mri_data_format, external_affine, use_tkr
):
    nibabel = pytest.importorskip("nibabel")

    # Generate rotation and scaling matrix equivalent to a unit cube
    rotation_matrix_3D = np.array(
        [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    )
    scale_matrix = 1.0 / M * np.identity(3)

    # Generate the affine mapping for nibabel
    A = np.append(np.dot(rotation_matrix_3D, scale_matrix), (translation).reshape(3, 1), axis=1)
    A = np.vstack([A, [0, 0, 0, 1]])
    Id = np.identity(4)

    # Write transformed data to file
    path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    if mri_data_format == "nifti":
        data = np.arange(1, M**3 + 1, dtype=np.float64).reshape(M, M, M)
        image = nibabel.nifti1.Nifti1Image(data, affine=Id if external_affine else A)
    elif mri_data_format == "mgh":
        data = np.arange(1, M**3 + 1, dtype=np.int32).reshape(M, M, M)
        image = nibabel.freesurfer.mghformat.MGHImage(data, affine=Id if external_affine else A)

    # Reorient the image to RAS
    orig_ornt = nibabel.io_orientation(image.affine)
    targ_ornt = nibabel.orientations.axcodes2ornt("RAS")
    transform = nibabel.orientations.ornt_transform(orig_ornt, targ_ornt)
    image = image.as_reoriented(transform)
    filename = path.with_suffix(".mgz")

    # Save the image to file
    if MPI.COMM_WORLD.rank == 0:
        nibabel.save(image, filename)
    MPI.COMM_WORLD.Barrier()
    # Create unit cube
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD,
        Nx,
        Ny,
        Nz,
        cell_type=dolfinx.cpp.mesh.CellType.hexahedron,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )

    # Find voxel position on reference cube (done prior to mesh transformation)
    q_el = basix.ufl.quadrature_element(mesh.topology.cell_name(), degree=degree)
    V = dolfinx.fem.functionspace(mesh, q_el)
    dx = 1 / M
    coords = np.abs(V.tabulate_dof_coordinates())  # Remove rounding error
    # NOTE: Voxel coordinates are based on center of cell rather than corner
    reference_input_index = ((coords) // dx).astype(np.int32)
    reference_data = data[
        reference_input_index[:, 0], reference_input_index[:, 1], reference_input_index[:, 2]
    ]

    # Transform mesh into physical space
    # As the voxel coordinates has origin in the midpoint of the RAS voxel, we add 1/(2M) to
    # the translation to account for this in the mesh, such that for a non-translated geoemtry
    # (theta=0), (translation=0), the origin in physical spaces is at (-0.5,-0.5,-0.5) in RAS,
    # which corresponds to the corner of voxel (0,0,0)
    midpoint_shift = 1.0 / (2 * M) * np.ones(3)
    shifted_coords = mesh.geometry.x - midpoint_shift

    # Rotate, then translate geometry
    mesh.geometry.x[:] = np.dot(rotation_matrix_3D, shifted_coords.T).T + translation

    # # Read data to function
    if external_affine:
        func = read_mri_data_to_function(
            filename, mesh, degree=degree, dtype=np.float64, vox2ras_transform=A, use_tkr=use_tkr
        )
    else:
        func = read_mri_data_to_function(
            filename, mesh, degree=degree, dtype=np.float64, use_tkr=use_tkr
        )

    np.testing.assert_allclose(func.x.array, reference_data)

    if degree == 0:
        cell_map = mesh.topology.index_map(mesh.topology.dim)
        num_cells_local = cell_map.size_local + cell_map.num_ghosts
        # Pick every second cell
        cells = np.arange(num_cells_local, dtype=np.int32)[::2]
        if external_affine:
            tag = read_mri_data_to_tag(
                filename,
                mesh,
                edim=mesh.topology.dim,
                entities=cells,
                vox2ras_transform=A,
                use_tkr=use_tkr,
            )
        else:
            tag = read_mri_data_to_tag(
                filename, mesh, edim=mesh.topology.dim, entities=cells, use_tkr=use_tkr
            )
        np.testing.assert_allclose(tag.values, reference_data[::2])
