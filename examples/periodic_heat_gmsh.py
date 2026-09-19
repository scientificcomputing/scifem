# # Periodicity from a gmsh model
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# The {doc}`Poisson example <periodic_poisson>` found its periodic vertex
# pairs from the coordinates, by giving
# {py:func}`scifem.periodic.create_periodic_mesh` an indicator and a mapping function.
# gmsh already knows the pairs when the model is meshed periodically, so a ``.msh`` file
# carries them in its ``$Periodic`` section and nothing has to be rediscovered.
# {py:func}`scifem.periodic.read_periodic_mesh_from_msh` reads them.
#
# That route is worth preferring when it is available. There is no tolerance to tune and
# no risk of snapping onto the wrong vertex, and it expresses rotational and reflective
# periodicity, which a coordinate mapping can only do if you write the transform out by
# hand.
#
# On that mesh we run the heat equation from a localised source. Each step is an implicit
# Euler solve, and the mass term makes the operator positive definite, so unlike the
# Poisson problem on a torus there is no kernel, no compatibility condition and no
# Lagrange multiplier -- `preonly` with a MUMPS LU factorisation just solves it.

# +
from mpi4py import MPI

import gmsh
import numpy as np
import pyvista
import ufl

import dolfinx
import dolfinx.fem.petsc
from scifem import assemble_scalar, evaluate_function
from scifem.periodic import read_periodic_mesh_from_msh
from scifem.periodic import transfer_function_to_parent_mesh

# -

# ## Meshing a periodic model
#
# We mesh the unit square and mark it periodic in $x$ **only**, so the left and right
# sides are glued while the top and bottom stay real boundaries. The domain is a cylinder,
# not a torus.
#
# `setPeriodic` takes the curves that are *replaced* first and their partners second,
# along with the $4\times 4$ affine transform, row-major, that carries the partner onto
# the replaced curve. Here that is a translation by one in $x$. The curve tags of
# `addRectangle` are 1 bottom, 2 right, 3 top and 4 left, so the right curve is replaced
# by the left.

# +
comm = MPI.COMM_WORLD
mesh_file = "periodic_cylinder.msh"
resolution = 0.025

if comm.rank == 0:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("periodic_cylinder")
    gmsh.model.occ.addRectangle(0, 0, 0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    # The reader takes the cells from the physical groups, so the surface needs one.
    gmsh.model.addPhysicalGroup(2, [s[1] for s in gmsh.model.getEntities(2)], tag=1)
    gmsh.model.mesh.setPeriodic(1, [2], [4], [1, 0, 0, 1.0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    gmsh.option.setNumber("Mesh.MeshSizeMin", resolution)
    gmsh.option.setNumber("Mesh.MeshSizeMax", resolution)
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_file)
    gmsh.finalize()
comm.Barrier()
# -

# Reading the file gives the periodic mesh directly. The rebuild needs a layer of ghost
# cells across every interprocess facet, which `model_to_mesh` does not provide by
# default, so the ghost mode is passed through.

periodic_mesh, replaced_vertices, replacement_map = read_periodic_mesh_from_msh(
    mesh_file, comm, gdim=2, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)

# We also read the mesh *without* the periodicity. It is needed to visualise the result
# on, and it lets us run the same problem with the seam left open, as a control. Reading
# the same file through the ordinary DOLFINx reader gives the same cells in the same
# order, which is what
# {py:func}`scifem.periodic.transfer_function_to_parent_mesh` requires.

mesh_data = dolfinx.io.gmsh.read_from_msh(
    mesh_file, comm, gdim=2, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)
mesh = getattr(mesh_data, "mesh", mesh_data)

if comm.rank == 0:
    print(
        f"vertices: {mesh.topology.index_map(0).size_global} -> "
        f"{periodic_mesh.topology.index_map(0).size_global}"
    )

# ## A heat source next to the seam
#
# Heat the cylinder from a Gaussian blob and let it spread:
#
# $$
# \frac{\partial u}{\partial t} = \Delta u + g,
# \qquad
# g(x, y) = \exp\!\left(-\frac{(x - x_0)^2 + (y - y_0)^2}{2\sigma^2}\right),
# \qquad u(0) = 0,
# $$
#
# with no flux through $y=0$ and $y=1$, which is the natural condition and so needs no
# code. Backward Euler with step $\Delta t$ gives, at each step,
#
# $$
# \int_\Omega u^{n+1} v \,\mathrm{d}x
#   + \Delta t \int_\Omega \nabla u^{n+1} \cdot \nabla v \,\mathrm{d}x
#   = \int_\Omega u^{n} v \,\mathrm{d}x
#   + \Delta t \int_\Omega g v \,\mathrm{d}x .
# $$
#
# The source sits at $x_0 = 0.15$, close to the seam and **away from the middle of the
# mesh**. That is the whole point of the placement. The heat has two ways round the
# cylinder, a short one of length $x_0$ through the seam and a long one of length
# $1 - x_0$, and the solution must end up symmetric about the line $x = x_0$ *measured
# around the cylinder*. A source at $x_0 = 0.5$ would sit on the mirror line of the square
# itself, so that symmetry would hold on an ordinary mesh too and would test nothing.

# +
x0, y0, sigma = 0.15, 0.5, 0.05
dt_value = 5e-3
num_steps = 20


def solve_heat(domain, degree=2):
    """Run the heat equation on `domain` from rest, and return the final temperature."""
    V = dolfinx.fem.functionspace(domain, ("Lagrange", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_n = dolfinx.fem.Function(V, name="u")

    x = ufl.SpatialCoordinate(domain)
    g = ufl.exp(-((x[0] - x0) ** 2 + (x[1] - y0) ** 2) / (2 * sigma**2))
    dt = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(dt_value))

    a = ufl.inner(u, v) * ufl.dx + dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(g, v) * ufl.dx

    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        u=u_n,  # solve in place, so the next step starts from this one
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
        petsc_options_prefix=f"heat_{id(domain)}_",
    )
    for _ in range(num_steps):
        problem.solve()
    return u_n, assemble_scalar(g * ufl.dx)


uh, source_integral = solve_heat(periodic_mesh)
# -

# ## Verification
#
# ### Nothing leaks
#
# Testing the step equation with $v = 1$, which the space contains, leaves
# $\int u^{n+1} = \int u^{n} + \Delta t \int g$: the stiffness term drops out because
# there is no flux anywhere, through the insulated sides or across the seam. After $n$
# steps the total heat is therefore exactly $n \Delta t \int_\Omega g$.

# +
total_heat = assemble_scalar(uh * ufl.dx)
expected = num_steps * dt_value * source_integral
if comm.rank == 0:
    print(
        f"int(u) dx = {total_heat:.8f}, expected {expected:.8f}, "
        f"relative error {abs(total_heat - expected) / expected:.2e}"
    )
# -

# ### The heat goes both ways round
#
# Now the test the off-centre source buys us. Sample the temperature at $x_0 \pm d$,
# wrapping around, along the line $y = y_0$. On the cylinder the two must agree; on the
# ordinary square the short way round is blocked, so they cannot.


# +
def line(xs):
    """Points at `xs` along y = y0, wrapped into the unit square."""
    return np.column_stack([xs % 1.0, np.full_like(xs, y0)])


offsets = np.array([0.1, 0.2, 0.3, 0.4])
control, _ = solve_heat(mesh)

# `evaluate_function` locates each point on whichever rank owns it and broadcasts the
# result, so the samples are available everywhere.
sampled = {
    label: (
        evaluate_function(field, line(x0 - offsets))[:, 0],
        evaluate_function(field, line(x0 + offsets))[:, 0],
    )
    for label, field in (("periodic", uh), ("plain", control))
}

if comm.rank == 0:
    print(f"\n{'d':>6} {'u(x0-d)':>10} {'u(x0+d)':>10} {'diff':>10} {'plain diff':>12}")
    for i, d in enumerate(offsets):
        behind, ahead = sampled["periodic"][0][i], sampled["periodic"][1][i]
        plain_behind, plain_ahead = sampled["plain"][0][i], sampled["plain"][1][i]
        print(
            f"{d:6.1f} {behind:10.6f} {ahead:10.6f} {abs(behind - ahead):10.2e}"
            f" {abs(plain_behind - plain_ahead):12.2e}"
        )
# -

# The periodic column agrees to discretisation error; the plain mesh is off by orders of
# magnitude, because on it $x_0 - d$ is a cold corner of the domain while $x_0 + d$ is
# downstream of the source.

# ## Visualisation
#
# The solution has to be moved onto the non-periodic mesh before it can be written or
# plotted: the writers place one output point per degree of freedom, and a degree of
# freedom on the seam has no single coordinate to be placed at.

# +
u_parent = transfer_function_to_parent_mesh(uh, mesh)

with dolfinx.io.VTXWriter(mesh.comm, "periodic_heat.bp", [u_parent]) as writer:
    writer.write(num_steps * dt_value)
# -

# Warped by its own value, the plume is a single smooth hill centred on the source, and it
# runs off the left edge of the square and back in on the right -- one hill on the
# cylinder, cut in two by where we chose to unroll it.

# +
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_parent.function_space))
grid.point_data["u"] = u_parent.x.array.real

# The temperature is small in absolute terms, so scale the warp to a fixed height.
peak = comm.allreduce(np.abs(u_parent.x.array).max(), op=MPI.MAX)

plotter = pyvista.Plotter()
plotter.add_mesh(
    grid.warp_by_scalar("u", factor=0.5 / peak),
    scalars="u",
    cmap="inferno",
    scalar_bar_args={"title": "u"},
)
plotter.add_mesh(grid, style="wireframe", color="gray", opacity=0.25)
plotter.camera_position = [(0.5, -2.1, 1.9), (0.5, 0.5, 0.15), (0.0, 0.0, 1.0)]
plotter.camera.zoom(1.15)
if not pyvista.OFF_SCREEN:
    plotter.show()
# -
