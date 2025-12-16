from mpi4py import MPI
from packaging.version import Version
import dolfinx
import scifem.interpolation
import pytest
import ufl
import numpy as np
import basix.ufl


@pytest.mark.skipif(
    np.issubdtype(dolfinx.default_scalar_type, np.complexfloating), reason="No complex support"
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
            f_rep[i] += np.pi * (i+1)
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
    np.issubdtype(dolfinx.default_scalar_type, np.complexfloating), reason="No complex support"
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
    np.issubdtype(dolfinx.default_scalar_type, np.complexfloating), reason="No complex support"
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
@pytest.mark.parametrize("family", ["Lagrange"])
@pytest.mark.skipif(
    Version(dolfinx.__version__) < Version("0.10.0"), reason="Requires DOLFINx version >0.10.0"
)
def test_interpolate_to_interface_submesh(family, degree):
    # Create a unit square
    comm = MPI.COMM_WORLD
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
    gamma_to_parent_map = gamma_to_parent.sub_topology_to_topology(
        np.arange(num_facets_local, dtype=np.int32), inverse=False
    )

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
        ue, qe, np.arange(len(gamma_to_parent_map), dtype=np.int32), mapped_entities[:, :2]
    )
    qe.x.scatter_forward()
    scifem.interpolation.interpolate_to_surface_submesh(
        ui, qi, np.arange(len(gamma_to_parent_map), dtype=np.int32), mapped_entities[:, 2:]
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
