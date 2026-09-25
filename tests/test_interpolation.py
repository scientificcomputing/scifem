from mpi4py import MPI
from packaging.version import Version
import dolfinx
import scifem.compat
import scifem.interpolation
import scifem.mesh
import pytest
import ufl
import numpy as np
import basix
import basix.ufl


@pytest.mark.skipif(
    np.issubdtype(dolfinx.default_scalar_type, np.complexfloating),
    reason="No complex support",
)
@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("out_family", ["Lagrange", "DG", "Quadrature"])
@pytest.mark.parametrize("value_shape", [(), (2,), (2, 3)])
def test_interpolation_matrix(use_petsc, cell_type, degree, out_family, value_shape):
    if use_petsc:
        pytest.importorskip("petsc4py")

    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type=cell_type)
    else:
        raise ValueError("Unsupported cell type")

    V = dolfinx.fem.functionspace(mesh, ("DG", degree, value_shape))
    if out_family == "Quadrature":
        el = basix.ufl.quadrature_element(mesh.basix_cell(), degree=degree, value_shape=value_shape)
    else:
        el = (out_family, degree, value_shape)
    Q = dolfinx.fem.functionspace(mesh, el)

    def f(x):
        scalar_val = x[0] ** degree + x[1] if tdim == 2 else x[0] + x[1] + x[2] ** degree
        vs = int(np.prod(value_shape))
        f_rep = np.tile(scalar_val, vs).reshape(vs, -1)
        for i in range(vs):
            f_rep[i] += np.pi * (i + 1)
        return f_rep

    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    q = dolfinx.fem.Function(Q)
    expr = ufl.TrialFunction(V)

    if use_petsc:
        A = scifem.interpolation.petsc_interpolation_matrix(expr, Q)
        A.mult(u.x.petsc_vec, q.x.petsc_vec)
        A.destroy()
    else:
        A = scifem.interpolation.interpolation_matrix(expr, Q)
        # Built in matrices has to use a special input vector, with additional ghosts.
        _x = dolfinx.la.vector(A.index_map(1), A.block_size[1])
        num_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        _x.array[:num_owned_dofs] = u.x.array[:num_owned_dofs]
        _x.scatter_forward()
        if not hasattr(dolfinx.la.MatrixCSR, "mult"):
            pytest.skip("MatrixCSR has no mult method")
        A.mult(_x, q.x)

    q.x.scatter_forward()

    q_ref = dolfinx.fem.Function(Q)
    if out_family == "Quadrature":
        try:
            ip = Q.element.interpolation_points()
        except TypeError:
            ip = Q.element.interpolation_points
        u_expr = dolfinx.fem.Expression(u, ip)
        q_ref.interpolate(u_expr)
    else:
        q_ref.interpolate(u)

    np.testing.assert_allclose(q.x.array, q_ref.x.array, rtol=1e-12, atol=1e-13)


@pytest.mark.skipif(
    np.issubdtype(dolfinx.default_scalar_type, np.complexfloating),
    reason="No complex support",
)
@pytest.mark.skipif(
    not hasattr(dolfinx.fem, "discrete_gradient"),
    reason="Cannot verify without discrete gradient from DOLFINx",
)
@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("degree", [1, 3, 5])
def test_discrete_gradient(degree, use_petsc, cell_type):
    if use_petsc:
        pytest.importorskip("petsc4py")

    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type=cell_type)
    else:
        raise ValueError("Unsupported cell type")

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    W = dolfinx.fem.functionspace(mesh, ("Nedelec 1st kind H(curl)", degree))

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] ** degree + x[1])

    w = dolfinx.fem.Function(W)
    expr = ufl.grad(ufl.TrialFunction(V))

    G_ref = dolfinx.fem.discrete_gradient(V, W)

    # Built in matrices has to use a special input vector, with additional ghosts.
    try:
        _x = dolfinx.la.vector(G_ref.index_map(1), G_ref.block_size[1])
    except AttributeError:
        # Bug in DOLFINx 0.9.0
        _x = dolfinx.la.vector(G_ref.index_map(1), G_ref.bs[1])

    num_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    _x.array[:num_owned_dofs] = u.x.array[:num_owned_dofs]
    _x.scatter_forward()

    if use_petsc:
        A = scifem.interpolation.petsc_interpolation_matrix(expr, W)
        A.mult(u.x.petsc_vec, w.x.petsc_vec)
        A.destroy()
    else:
        if not hasattr(dolfinx.la.MatrixCSR, "mult"):
            pytest.skip("MatrixCSR has no mult method")
        A = scifem.interpolation.interpolation_matrix(expr, W)
        A.mult(_x, w.x)
    w.x.scatter_forward()

    w_ref = dolfinx.fem.Function(W)
    if not hasattr(dolfinx.la.MatrixCSR, "mult"):
        # Fallback to PETSc discrete gradient on 0.9
        pytest.mark.skipif(not dolfinx.has_petsc4py, reason="Cannot verify without petsc4py")
        import dolfinx.fem.petsc as _petsc

        G_ref = _petsc.discrete_gradient(V, W)
        G_ref.assemble()
        G_ref.mult(u.x.petsc_vec, w_ref.x.petsc_vec)
    else:
        G_ref.mult(_x, w_ref.x)
    w_ref.x.scatter_forward()

    np.testing.assert_allclose(w.x.array, w_ref.x.array, rtol=1e-11, atol=1e-12)


@pytest.mark.skipif(
    np.issubdtype(dolfinx.default_scalar_type, np.complexfloating),
    reason="No complex support",
)
@pytest.mark.skipif(
    not hasattr(dolfinx.fem, "discrete_curl"),
    reason="Cannot verify without discrete curl from DOLFINx",
)
@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("use_petsc", [True, False])
@pytest.mark.parametrize("degree", [1, 2])
def test_discrete_curl(degree, use_petsc, cell_type):
    if use_petsc:
        pytest.importorskip("petsc4py")

    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, cell_type=cell_type)
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type=cell_type)
    else:
        raise ValueError("Unsupported cell type")

    V = dolfinx.fem.functionspace(mesh, ("Nedelec 2nd kind H(curl)", degree + 1))
    W = dolfinx.fem.functionspace(mesh, ("RT", degree))

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: (x[0] ** degree, x[1] ** degree - 1, -x[2]))

    w = dolfinx.fem.Function(W)
    expr = ufl.curl(ufl.TrialFunction(V))

    G_ref = dolfinx.fem.discrete_curl(V, W)

    # Built in matrices has to use a special input vector, with additional ghosts.
    _x = dolfinx.la.vector(G_ref.index_map(1), G_ref.block_size[1])
    num_owned_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    _x.array[:num_owned_dofs] = u.x.array[:num_owned_dofs]
    _x.scatter_forward()

    if use_petsc:
        A = scifem.interpolation.petsc_interpolation_matrix(expr, W)
        A.mult(u.x.petsc_vec, w.x.petsc_vec)
        A.destroy()
    else:
        if not hasattr(dolfinx.la.MatrixCSR, "mult"):
            pytest.skip("MatrixCSR has no mult method")
        A = scifem.interpolation.interpolation_matrix(expr, W)
        A.mult(_x, w.x)
    w.x.scatter_forward()

    w_ref = dolfinx.fem.Function(W)
    G_ref.mult(_x, w_ref.x)
    w_ref.x.scatter_forward()

    np.testing.assert_allclose(w.x.array, w_ref.x.array, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("family", ["Lagrange", "DG"])
@pytest.mark.skipif(
    Version(dolfinx.__version__) < Version("0.10.0"),
    reason="Requires DOLFINx version >0.10.0",
)
def test_interpolate_to_interface_submesh(family, degree):
    # Create a unit square
    comm = MPI.COMM_WORLD

    if Version(dolfinx.__version__) < Version("0.11.0.dev0") and family == "DG":
        pytest.skip("Interpolation to surface submesh does not work for DG in DOLFINx < 0.11.0")
    domain = dolfinx.mesh.create_unit_square(
        comm, 48, 48, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )

    # Split unit square in two subdomains
    cell_map = domain.topology.index_map(domain.topology.dim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts
    markers = np.full(num_cells_local, 1, dtype=np.int32)
    markers[
        dolfinx.mesh.locate_entities(domain, domain.topology.dim, lambda x: x[0] <= 0.5 + 1e-14)
    ] = 2
    ct = dolfinx.mesh.meshtags(
        domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32), markers
    )

    # Create submesh for each subdomain
    omega_e, e_to_parent, _, _, _ = scifem.mesh.extract_submesh(domain, ct, (1,))
    omega_i, i_to_parent, _, _, _ = scifem.mesh.extract_submesh(domain, ct, (2,))

    # Compute submesh for the interface between omega_e and omega_i
    interface_facets = scifem.mesh.find_interface(ct, (1,), (2,))
    ft = dolfinx.mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        interface_facets,
        np.full(interface_facets.shape, 1, dtype=np.int32),
    )

    gamma, gamma_to_parent, _, _, _ = scifem.mesh.extract_submesh(domain, ft, 1)

    num_facets_local = (
        gamma.topology.index_map(gamma.topology.dim).size_local
        + gamma.topology.index_map(gamma.topology.dim).num_ghosts
    )
    gamma_to_parent_map = scifem.mesh.get_entity_map(gamma_to_parent)[:num_facets_local]

    # Create functions on each subdomain
    def fe(x):
        return x[0] + x[1] ** degree

    def fi(x):
        return np.sin(x[0]) + np.cos(x[1])

    Ve = dolfinx.fem.functionspace(omega_e, (family, degree))
    ue = dolfinx.fem.Function(Ve)
    ue.interpolate(fe)
    ue.x.scatter_forward()
    Vi = dolfinx.fem.functionspace(omega_i, (family, degree))
    ui = dolfinx.fem.Function(Vi)
    ui.interpolate(fi)
    ui.x.scatter_forward()

    # Compute ordered integration entities on the interface
    interface_integration_entities = scifem.compute_interface_data(
        ct, facet_indices=gamma_to_parent_map, include_ghosts=True
    )
    mapped_entities = interface_integration_entities.copy()

    # For each submesh, get the relevant integration entities
    parent_to_e = e_to_parent.sub_topology_to_topology(
        np.arange(num_cells_local, dtype=np.int32), inverse=True
    )
    parent_to_i = i_to_parent.sub_topology_to_topology(
        np.arange(num_cells_local, dtype=np.int32), inverse=True
    )
    mapped_entities[:, 0] = parent_to_e[interface_integration_entities[:, 0]]
    mapped_entities[:, 2] = parent_to_i[interface_integration_entities[:, 2]]
    assert np.all(mapped_entities[:, 0] >= 0)
    assert np.all(mapped_entities[:, 2] >= 0)

    # Create two functions on the interface submesh
    Q = dolfinx.fem.functionspace(gamma, (family, degree))
    qe = dolfinx.fem.Function(Q, name="qe")
    qi = dolfinx.fem.Function(Q, name="qi")

    # Interpolate volume functions (on submesh) onto all cells of the interface submesh
    scifem.interpolation.interpolate_to_surface_submesh(
        ue,
        qe,
        np.arange(len(gamma_to_parent_map), dtype=np.int32),
        mapped_entities[:, :2],
    )
    qe.x.scatter_forward()
    scifem.interpolation.interpolate_to_surface_submesh(
        ui,
        qi,
        np.arange(len(gamma_to_parent_map), dtype=np.int32),
        mapped_entities[:, 2:],
    )
    qi.x.scatter_forward()

    # Compute the difference between the two interpolated functions
    I = dolfinx.fem.Function(Q, name="i")
    I.x.array[:] = qe.x.array - qi.x.array

    reference = dolfinx.fem.Function(Q)
    reference.interpolate(lambda x: fe(x) - fi(x))

    qe_ref = dolfinx.fem.Function(Q)
    qe_ref.interpolate(fe)
    qi_ref = dolfinx.fem.Function(Q)
    qi_ref.interpolate(fi)
    np.testing.assert_allclose(qe.x.array, qe_ref.x.array)
    np.testing.assert_allclose(qi.x.array, qi_ref.x.array)
    np.testing.assert_allclose(I.x.array, reference.x.array, rtol=1e-14, atol=1e-14)


def _exterior_facet_submesh(mesh):
    """The exterior facets as a submesh, its cells, their parent integration entities, and the
    submesh's entity map."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, fdim + 1)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    submesh, entity_map = dolfinx.mesh.create_submesh(mesh, fdim, facets)[:2]
    num_facets = submesh.topology.index_map(fdim).size_local
    submesh_facets = np.arange(num_facets, dtype=np.int32)
    parent_facets = scifem.mesh.get_entity_map(entity_map)[:num_facets]
    entities = scifem.compat.compute_integration_domains(
        dolfinx.fem.IntegralType.exterior_facet, mesh.topology, parent_facets
    ).reshape(-1, 2)
    return submesh, submesh_facets, entities, entity_map


_EXTENSION_MESHES = {
    "triangle": lambda: dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 3),
    "quadrilateral": lambda: dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 4, 3, cell_type=dolfinx.mesh.CellType.quadrilateral
    ),
    "tetrahedron": lambda: dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2),
    "hexahedron": lambda: dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, 2, 2, 2, cell_type=dolfinx.mesh.CellType.hexahedron
    ),
}


def _surface_values(x):
    return np.vstack((np.sin(2.0 * x[0]) + x[1] ** 3, x[0] * x[-1] + 1.0))


@pytest.mark.skipif(
    Version(dolfinx.__version__) < Version("0.10.0"), reason="Requires DOLFINx >= 0.10"
)
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("cell", sorted(_EXTENSION_MESHES))
def test_interpolate_from_surface_submesh(cell, degree):
    """The extension puts the surface values on the boundary, zero elsewhere, and restricting it
    returns the input. From P3 an edge carries two dofs and a hexahedron's face four, whose order
    depends on the facet's orientation in its cell."""
    mesh = _EXTENSION_MESHES[cell]()
    submesh, submesh_facets, entities, _ = _exterior_facet_submesh(mesh)
    V_parent = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (2,)))
    V_submesh = dolfinx.fem.functionspace(submesh, ("Lagrange", degree, (2,)))
    u_submesh = dolfinx.fem.Function(V_submesh)
    u_submesh.interpolate(_surface_values)
    u_parent = dolfinx.fem.Function(V_parent)
    scifem.interpolation.interpolate_from_surface_submesh(
        u_submesh, u_parent, submesh_facets, entities
    )

    # Every dof not on the facet submesh must still be zero
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V_parent, mesh.topology.dim - 1, dolfinx.mesh.exterior_facet_indices(mesh.topology)
    )
    interior_dofs = np.setdiff1d(np.arange(V_parent.dofmap.index_map.size_local), boundary_dofs)
    assert np.all(u_parent.x.array.reshape(-1, 2)[interior_dofs] == 0.0)

    # Round trip: restricting the extension back to the submesh returns the input
    u_restricted = dolfinx.fem.Function(V_submesh)
    scifem.interpolation.interpolate_to_surface_submesh(
        u_parent, u_restricted, submesh_facets, entities
    )
    np.testing.assert_allclose(u_restricted.x.array, u_submesh.x.array, atol=1e-13)


@pytest.mark.skipif(
    Version(dolfinx.__version__) < Version("0.10.0"), reason="Requires DOLFINx >= 0.10"
)
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("cell", ["triangle", "tetrahedron"])
def test_interpolate_from_a_coarser_surface_space(cell, degree):
    """A degree P surface function into a degree P+1 volume space gives its interpolant at the
    boundary nodes."""
    mesh = _EXTENSION_MESHES[cell]()
    submesh, submesh_facets, entities, _ = _exterior_facet_submesh(mesh)
    coarse_element = ("Lagrange", degree, (2,))
    fine_element = ("Lagrange", degree + 1, (2,))
    u_submesh = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, coarse_element))
    u_submesh.interpolate(_surface_values)
    u_parent = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, fine_element))
    scifem.interpolation.interpolate_from_surface_submesh(
        u_submesh, u_parent, submesh_facets, entities
    )

    # The same interpolant, built on the submesh first in the volume's element
    u_submesh_fine = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, fine_element))
    u_submesh_fine.interpolate(u_submesh)
    u_expected = dolfinx.fem.Function(u_parent.function_space)
    scifem.interpolation.interpolate_from_surface_submesh(
        u_submesh_fine, u_expected, submesh_facets, entities
    )
    np.testing.assert_allclose(u_parent.x.array, u_expected.x.array, atol=1e-13)


def test_interpolate_from_a_discontinuous_surface_space_averages_shared_nodes():
    """At a node shared by facets that disagree, the extension takes the mean of their values."""
    mesh = _EXTENSION_MESHES["triangle"]()
    submesh, submesh_facets, entities, _ = _exterior_facet_submesh(mesh)
    V_submesh = dolfinx.fem.functionspace(submesh, ("DG", 0))
    u_submesh = dolfinx.fem.Function(V_submesh)
    u_submesh.x.array[:] = np.arange(u_submesh.x.array.size, dtype=u_submesh.x.array.dtype)
    V_parent = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u_parent = dolfinx.fem.Function(V_parent)
    scifem.interpolation.interpolate_from_surface_submesh(
        u_submesh, u_parent, submesh_facets, entities
    )

    # The same means, computed facet by facet: each facet's value at its closure dofs
    fdim = mesh.topology.dim - 1
    closure_dofs = scifem.interpolation.compute_entity_closure_dofs(V_parent, fdim, entities)
    facet_values = np.repeat(
        u_submesh.x.array[V_submesh.dofmap.list[submesh_facets, 0]], closure_dofs.shape[1]
    )
    summed_values = dolfinx.fem.Function(V_parent)
    num_writes = dolfinx.fem.Function(V_parent)
    np.add.at(summed_values.x.array, closure_dofs.ravel(), facet_values)
    np.add.at(num_writes.x.array, closure_dofs.ravel(), 1.0)
    for function in (summed_values, num_writes):
        function.x.scatter_reverse(dolfinx.la.InsertMode.add)
        function.x.scatter_forward()
    is_written = num_writes.x.array > 0
    u_expected = summed_values.x.array[is_written] / num_writes.x.array[is_written]
    np.testing.assert_allclose(u_parent.x.array[is_written], u_expected, atol=1e-13)
    assert np.all(u_parent.x.array[~is_written] == 0.0)


def _linear_field(x):
    """A linear vector field, exactly represented by vector P1 on the submesh."""
    return np.vstack([1.0 + x[0] - 0.5 * x[1] + (k + 1) * x[k] for k in range(x.shape[0])])


_PIOLA_ELEMENTS = [("RT", 1), ("RT", 2), ("N1curl", 1), ("N1curl", 2)]


@pytest.mark.skipif(
    Version(dolfinx.__version__) < Version("0.10.0"), reason="Requires DOLFINx >= 0.10"
)
@pytest.mark.parametrize("element", _PIOLA_ELEMENTS, ids=[f"{f}{d}" for f, d in _PIOLA_ELEMENTS])
@pytest.mark.parametrize("cell", sorted(_EXTENSION_MESHES))
def test_interpolate_from_surface_submesh_into_a_piola_mapped_space(cell, element):
    """The extension into an H(div) or H(curl) space gives the dofs that interpolating the same
    field into that space gives on the facets' closures, and zero everywhere else."""
    mesh = _EXTENSION_MESHES[cell]()
    submesh, submesh_facets, entities, entity_map = _exterior_facet_submesh(mesh)
    gdim = mesh.geometry.dim
    u_submesh = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, ("Lagrange", 1, (gdim,))))
    u_submesh.interpolate(lambda x: _linear_field(x[:gdim]))
    V_parent = dolfinx.fem.functionspace(mesh, element)
    u_parent = dolfinx.fem.Function(V_parent)
    scifem.interpolation.interpolate_from_surface_submesh(
        u_submesh, u_parent, submesh_facets, entities, entity_maps=[entity_map]
    )

    # The reference: the same field interpolated into the whole space, kept on the facets' closures
    u_interpolated = dolfinx.fem.Function(V_parent)
    u_interpolated.interpolate(lambda x: _linear_field(x[:gdim]))
    fdim = mesh.topology.dim - 1
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V_parent, fdim, dolfinx.mesh.exterior_facet_indices(mesh.topology)
    )
    u_expected = np.zeros_like(u_parent.x.array)
    u_expected[boundary_dofs] = u_interpolated.x.array[boundary_dofs]
    np.testing.assert_allclose(u_parent.x.array, u_expected, atol=1e-12)
    assert np.abs(u_expected).max() > 0.1


def test_interpolate_from_surface_submesh_into_a_piola_mapped_space_needs_the_entity_map():
    """The surface function is evaluated on the parent mesh's facets, through the entity map."""
    mesh = _EXTENSION_MESHES["triangle"]()
    submesh, submesh_facets, entities, _ = _exterior_facet_submesh(mesh)
    u_submesh = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, ("Lagrange", 1, (2,))))
    u_parent = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("N1curl", 1)))
    with pytest.raises(ValueError, match="needs entity_maps"):
        scifem.interpolation.interpolate_from_surface_submesh(
            u_submesh, u_parent, submesh_facets, entities
        )


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("cell", sorted(_EXTENSION_MESHES))
def test_facet_closure_permutations_match_a_per_facet_loop(cell, degree):
    """Grouping by (cell permutation info, local facet) gives each facet the permutation that
    basix computes for it on its own."""
    mesh = _EXTENSION_MESHES[cell]()
    _, _, entities, _ = _exterior_facet_submesh(mesh)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    fdim = mesh.topology.dim - 1
    permutations = scifem.interpolation.compute_entity_closure_permutations(V, fdim, entities)
    closure_dofs = scifem.interpolation.compute_entity_closure_dofs(V, fdim, entities)

    cell_info = mesh.topology.get_cell_permutation_info()
    facet_types = basix.cell.subentity_types(V.element.basix_element.cell_type)[fdim]
    layout = V.dofmap.dof_layout
    for i, (cell_index, local_facet) in enumerate(entities):
        expected_permutation = np.arange(permutations.shape[1], dtype=np.int32)
        V.element.basix_element.permute_subentity_closure_inv(
            expected_permutation,
            int(cell_info[cell_index]),
            facet_types[local_facet],
            int(local_facet),
        )
        np.testing.assert_array_equal(permutations[i], expected_permutation)
        closure = np.asarray(layout.entity_closure_dofs(fdim, int(local_facet)))
        expected_closure_dofs = V.dofmap.list[cell_index][closure[expected_permutation]]
        np.testing.assert_array_equal(closure_dofs[i], expected_closure_dofs)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("cell", ["tetrahedron", "hexahedron", "triangle", "quadrilateral"])
def test_entity_closure_dofs_agree_between_cells(cell, degree):
    """Every cell sharing an edge or a facet gives the same ordered closure dofs for it.

    From P3 an edge carries two interior dofs and a hexahedron's face four, so this pins the
    orientation of each entity, not only which dofs its closure holds.
    """
    mesh = _EXTENSION_MESHES[cell]()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    for dim in range(1, tdim):
        mesh.topology.create_connectivity(tdim, dim)
        cell_to_entity = mesh.topology.connectivity(tdim, dim)
        entities_per_cell = dolfinx.cpp.mesh.cell_num_entities(mesh.topology.cell_type, dim)
        cells = np.repeat(np.arange(num_cells, dtype=np.int32), entities_per_cell)
        local_entities = np.tile(np.arange(entities_per_cell, dtype=np.int32), num_cells)
        closure_dofs = scifem.interpolation.compute_entity_closure_dofs(
            V, dim, np.column_stack((cells, local_entities))
        )
        entity_indices = cell_to_entity.array.reshape(-1)
        closure_dofs_of_entity: dict[int, np.ndarray] = {}
        for entity, closure in zip(entity_indices, closure_dofs):
            expected_closure = closure_dofs_of_entity.setdefault(int(entity), closure)
            np.testing.assert_array_equal(closure, expected_closure)


@pytest.mark.skipif(
    Version(dolfinx.__version__) < Version("0.10.0"), reason="Requires DOLFINx >= 0.10"
)
@pytest.mark.parametrize("cell", ["triangle", "hexahedron"])
def test_surface_submesh_interpolation_is_reusable(cell):
    """One prepared interpolation follows later changes to its coefficients' values."""
    mesh = _EXTENSION_MESHES[cell]()
    submesh, submesh_facets, entities, _ = _exterior_facet_submesh(mesh)
    V_parent = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    V_submesh = dolfinx.fem.functionspace(submesh, ("Lagrange", 2, (2,)))
    u_parent = dolfinx.fem.Function(V_parent)
    interpolation = scifem.interpolation.SurfaceSubmeshInterpolation(
        u_parent, V_submesh, submesh_facets, entities
    )
    for scale in (1.0, -2.5):
        u_parent.interpolate(lambda x: scale * _surface_values(x))
        u_submesh = dolfinx.fem.Function(V_submesh)
        interpolation.apply(u_submesh)
        u_expected = dolfinx.fem.Function(V_submesh)
        scifem.interpolation.interpolate_to_surface_submesh(
            u_parent, u_expected, submesh_facets, entities
        )
        np.testing.assert_allclose(u_submesh.x.array, u_expected.x.array, atol=1e-14)
        assert np.abs(u_submesh.x.array).max() > 0.1
