from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import basix
import ufl
import numpy as np
import scifem
import pytest
from petsc4py import PETSc
from unittest.mock import patch

try:
    import h5py  # noqa: F401

    hash5py = True
except ImportError:
    hash5py = False


stype = PETSc.ScalarType
rtype = PETSc.RealType


@pytest.mark.parametrize("use_ctx_manager", [True, False])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("h5py", marks=pytest.mark.skipif(not hash5py, reason="h5py is not installed")),
        "adios2",
    ],
)
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
@pytest.mark.parametrize("value_shape", [(), (2,)])
def test_XDMFFile_2D(cell_type, degree, value_shape, backend, use_ctx_manager, tmp_path):
    folder = MPI.COMM_WORLD.bcast(tmp_path, root=0)

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type, dtype=np.float64)

    el = basix.ufl.quadrature_element(
        scheme="default", degree=degree, cell=mesh.ufl_cell().cellname(), value_shape=value_shape
    )
    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V, dtype=stype)
    v = dolfinx.fem.Function(V, dtype=stype)
    t = dolfinx.fem.Constant(mesh, stype(0.0))

    X = ufl.SpatialCoordinate(mesh)
    if value_shape == ():
        u_expr = ufl.sin(ufl.pi * X[0]) * ufl.sin(ufl.pi * X[1]) * ufl.cos(ufl.pi * t)
        v_expr = ufl.cos(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]) * ufl.cos(ufl.pi * t)

    else:
        u_expr = X * ufl.cos(ufl.pi * t)
        v_expr = X * ufl.sin(X[0]) * ufl.cos(ufl.pi * t)

    u.name = "u"
    v.name = "v"

    if use_ctx_manager:
        with scifem.xdmf.XDMFFile(folder / "data.xdmf", [u, v], backend=backend) as xdmf:
            u.interpolate(dolfinx.fem.Expression(u_expr, V.element.interpolation_points()))
            v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
            xdmf.write(rtype(t.value))

            t.value = 0.3
            u.interpolate(dolfinx.fem.Expression(u_expr, V.element.interpolation_points()))
            v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
            xdmf.write(rtype(t.value))
    else:
        xdmf = scifem.xdmf.XDMFFile(folder / "data.xdmf", [u, v], backend=backend)
        u.interpolate(dolfinx.fem.Expression(u_expr, V.element.interpolation_points()))
        v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
        xdmf.write(rtype(t.value))

        t.value = 0.3
        u.interpolate(dolfinx.fem.Expression(u_expr, V.element.interpolation_points()))
        v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
        xdmf.write(rtype(t.value))

        xdmf.close()

    assert (folder / "data.xdmf").is_file()
    assert (folder / "data.h5").is_file()


@pytest.mark.parametrize("use_ctx_manager", [True, False])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("h5py", marks=pytest.mark.skipif(not hash5py, reason="h5py is not installed")),
        "adios2",
    ],
)
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
@pytest.mark.parametrize("value_shape", [(), (3,)])
def test_XDMFFile_3D(cell_type, degree, value_shape, backend, use_ctx_manager, tmp_path):
    folder = MPI.COMM_WORLD.bcast(tmp_path, root=0)

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 3, 5, cell_type, dtype=np.float64)

    el = basix.ufl.quadrature_element(
        scheme="default", degree=degree, cell=mesh.ufl_cell().cellname(), value_shape=value_shape
    )
    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V, dtype=stype)
    v = dolfinx.fem.Function(V, dtype=stype)
    t = dolfinx.fem.Constant(mesh, stype(0.0))

    X = ufl.SpatialCoordinate(mesh)
    if value_shape == ():
        u_expr = ufl.sin(ufl.pi * X[0]) * ufl.sin(ufl.pi * X[1]) * ufl.cos(ufl.pi * t)
        v_expr = ufl.cos(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]) * ufl.cos(ufl.pi * t)

    else:
        u_expr = X * ufl.cos(ufl.pi * t)
        v_expr = X * ufl.sin(X[0]) * ufl.cos(ufl.pi * t)

    u.name = "u"
    v.name = "v"

    if use_ctx_manager:
        with scifem.xdmf.XDMFFile(folder / "data.xdmf", [u, v], backend=backend) as xdmf:
            u.interpolate(dolfinx.fem.Expression(u_expr, V.element.interpolation_points()))
            v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
            xdmf.write(rtype(t.value))

            t.value = 0.3
            u.interpolate(dolfinx.fem.Expression(u_expr, V.element.interpolation_points()))
            v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
            xdmf.write(rtype(t.value))
    else:
        xdmf = scifem.xdmf.XDMFFile(folder / "data.xdmf", [u, v], backend=backend)
        u.interpolate(dolfinx.fem.Expression(u_expr, V.element.interpolation_points()))
        v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
        xdmf.write(rtype(t.value))

        t.value = 0.3
        u.interpolate(dolfinx.fem.Expression(u_expr, V.element.interpolation_points()))
        v.interpolate(dolfinx.fem.Expression(v_expr, V.element.interpolation_points()))
        xdmf.write(rtype(t.value))

        xdmf.close()

    assert (folder / "data.xdmf").is_file()
    assert (folder / "data.h5").is_file()


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral]
)
@pytest.mark.parametrize("value_shape", [(), (2,)])
def test_create_ponitcloud_2D(cell_type, degree, value_shape, tmp_path):
    folder = MPI.COMM_WORLD.bcast(tmp_path, root=0)

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type, dtype=np.float64)

    el = basix.ufl.quadrature_element(
        scheme="default", degree=degree, cell=mesh.ufl_cell().cellname(), value_shape=value_shape
    )
    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    v = dolfinx.fem.Function(V)

    if value_shape == ():
        u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        v.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))
    else:
        u.interpolate(lambda x: x[:2, :])
        v.interpolate(lambda x: 2 * x[:2, :])
    u.name = "u"

    v.name = "v"

    scifem.xdmf.create_pointcloud(folder / "data.xdmf", [u, v])
    assert (folder / "data.xdmf").is_file()
    assert (folder / "data.h5").is_file()


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
@pytest.mark.parametrize("value_shape", [(), (3,)])
def test_create_ponitcloud_3D(cell_type, degree, value_shape, tmp_path):
    folder = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 3, 5, cell_type, dtype=np.float64)

    el = basix.ufl.quadrature_element(
        scheme="default", degree=degree, cell=mesh.ufl_cell().cellname(), value_shape=value_shape
    )
    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    v = dolfinx.fem.Function(V)

    if value_shape == ():
        u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2]))
        v.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2]))
    else:
        u.interpolate(lambda x: x[:3, :])
        v.interpolate(lambda x: 2 * x[:3, :])

    u.name = "u"
    v.name = "v"

    scifem.xdmf.create_pointcloud(folder / "data.xdmf", [u, v])
    assert (folder / "data.xdmf").is_file()
    assert (folder / "data.h5").is_file()


def test_different_function_spaces_raises_ValueError(tmp_path):
    folder = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, 2, 3, 5, dolfinx.mesh.CellType.tetrahedron, dtype=np.float64
    )
    V1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    V2 = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))

    with pytest.raises(ValueError):
        scifem.xdmf.create_pointcloud(
            folder / "data.xdmf", [dolfinx.fem.Function(V1), dolfinx.fem.Function(V2)]
        )


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
@pytest.mark.parametrize("value_shape", [(), (3,)])
@pytest.mark.skipif(not hash5py, reason="h5py not installed")
def test_h5py_fallback_3D(cell_type, degree, value_shape, tmp_path):
    folder = MPI.COMM_WORLD.bcast(tmp_path, root=0)

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 3, 5, cell_type, dtype=np.float64)

    el = basix.ufl.quadrature_element(
        scheme="default", degree=degree, cell=mesh.ufl_cell().cellname(), value_shape=value_shape
    )
    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V)
    v = dolfinx.fem.Function(V)

    if value_shape == ():
        u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2]))
        v.interpolate(lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2]))
    else:
        u.interpolate(lambda x: x[:3, :])
        v.interpolate(lambda x: 2 * x[:3, :])

    u.name = "u"
    v.name = "v"

    with patch.dict("sys.modules", {"adios2": None}):
        scifem.xdmf.create_pointcloud(folder / "data.xdmf", [u, v])
    assert (folder / "data.xdmf").is_file()
    assert (folder / "data.h5").is_file()
