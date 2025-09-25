from mpi4py import MPI
import dolfinx
from scifem import create_space_of_simple_functions, assemble_scalar
import pytest
import numpy as np
import ufl


@pytest.mark.parametrize("value_shape", [(), (3,), (3, 2)])
@pytest.mark.parametrize(
    "cell_type", ["interval", "triangle", "quadrilateral", "hexahedron", "tetrahedron"]
)
def test_material_space(value_shape: tuple[int], cell_type: str):
    cell = dolfinx.mesh.to_type(cell_type)
    tdim = dolfinx.mesh.cell_dim(cell)
    if tdim == 1:
        mesh = dolfinx.mesh.create_unit_interval(
            MPI.COMM_WORLD, 20, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
        )
    elif tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, 20, 20, ghost_mode=dolfinx.mesh.GhostMode.shared_facet, cell_type=cell
        )
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(
            MPI.COMM_WORLD,
            12,
            12,
            8,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            cell_type=cell,
        )
    else:
        raise ValueError(f"Unsupported {cell_type=}")

    def ball(x, tol=1e-8):
        conditions = []
        for i in range(tdim):
            conditions.append((x[i] - 0.5) ** 2 <= 0.2**2 + tol)
        return np.logical_and.reduce(conditions)

    def top(x, tol=1e-8):
        return x[tdim - 1] >= 0.7 - tol

    cell_map = mesh.topology.index_map(mesh.topology.dim)
    all_cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
    tags = [1, 5, 3]
    values = np.full_like(all_cells, tags[0])
    values[dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, ball)] = tags[1]
    values[dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, top)] = tags[2]
    ct = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, all_cells, values)

    V = create_space_of_simple_functions(mesh, ct, tags, value_shape=value_shape)
    assert V.dofmap.index_map.size_global == len(tags)
    assert V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts == len(tags)
    assert (V.dofmap.index_map.size_local == 0) or (V.dofmap.index_map.num_ghosts == 0)
    value_size = int(np.prod(value_shape))
    assert V.dofmap.index_map_bs == value_size
    assert V.dofmap.bs == value_size
    array = V.dofmap.list
    for i, tag in enumerate(tags):
        np.testing.assert_allclose(array[ct.indices[ct.values == tag]].flatten(), i)

    # Check that we can assign values to the function for each region, and that it gives
    # a sensible result
    bs = V.dofmap.bs
    for i, tag in enumerate(tags):
        vals = np.arange(i + 1, i + 1 + bs, dtype=np.float64)
        f = dolfinx.fem.Function(V)
        f.x.array[i * bs : (i + 1) * bs] = vals

        form = dolfinx.fem.form(ufl.inner(f, f) * ufl.dx)
        integral = assemble_scalar(form)

        # Compute integral of exact function on the restricted domain
        dx_restricted = ufl.Measure("dx", domain=mesh, subdomain_data=ct, subdomain_id=tag)
        if value_shape == ():
            f_ex = vals[0]
        else:
            f_ex = ufl.as_tensor(vals.reshape(value_shape))
        form_ex = ufl.inner(f_ex, f_ex) * dx_restricted
        integral_ex = assemble_scalar(form_ex)
        np.testing.assert_allclose(integral, integral_ex)

        # Compute squared error on the restricted domain
        diff = ufl.inner(f - f_ex, f - f_ex) * dx_restricted
        error = assemble_scalar(diff)
        np.testing.assert_allclose(error, 0.0)
