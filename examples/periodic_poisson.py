# # Poisson on a periodic mesh
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# This example solves the Poisson problem on a doubly periodic unit square built with
# {py:func}`scifem.periodic.create_periodic_mesh`, and covers the three things that
# are easy to get wrong on such a mesh:
#
# 1. **Building it.** What the indicator and mapping functions have to do, and what the
#    resulting mesh does and does not change.
# 2. **Checking the answer.** A fully periodic domain has no boundary, so the constants
#    are in the kernel and the source has to be mean free. Less obviously, most of the
#    natural test solutions also solve the *homogeneous Neumann* problem on the ordinary
#    square, so they pass on a mesh where periodicity is broken.
# 3. **Looking at it.** `VTXWriter` and `VTKFile` draw a periodic mesh wrong, for a
#    reason that is worth understanding rather than working around blindly.

# +
from mpi4py import MPI

import numpy as np
import pyvista
import ufl

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
from scifem import assemble_scalar
from scifem.periodic import create_periodic_mesh
from scifem.periodic import transfer_function_to_parent_mesh

# -

# ## Creating a periodic mesh
#
# We start from an ordinary mesh. It must carry a layer of ghost cells across every
# interprocess facet, which is the default for
# {py:func}`dolfinx.mesh.create_unit_square`; {py:func}`scifem.periodic.create_periodic_mesh` checks this and
# raises if it is missing.

N = 25
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

# Periodicity is described by two functions of a `(3, num_points)` coordinate array.
# The **indicator** marks the vertices that are to disappear, and the **mapping
# function** says, for each marked vertex, which vertex it is identified with. Here we
# remove the $x=1$ and $y=1$ sides and glue them onto $x=0$ and $y=0$.
#
# Both functions are evaluated on the same array, so they must be written to handle the
# corner $(1, 1)$ as well: it is marked once, and the mapping has to shift it in *both*
# directions at once so that it lands on $(0, 0)$. Writing the shift as a subtraction of
# the boolean mask, rather than as an `if`, does exactly that.


def indicator(x):
    return np.isclose(x[0], 1.0) | np.isclose(x[1], 1.0)


def mapping_function(x):
    values = x.copy()
    values[0] -= np.isclose(x[0], 1.0)
    values[1] -= np.isclose(x[1], 1.0)
    return values


periodic_mesh, replaced_vertices, replacement_map = create_periodic_mesh(
    mesh, indicator, mapping_function
)

# The rebuild is purely topological, and that distinction matters for everything below.
# The **topology** loses one row and one column of vertices, because each pair has been
# merged into a single vertex. The **geometry** is untouched: every node of the original
# mesh is still there, with its original coordinates, and the cells are the same cells.

tdim = periodic_mesh.topology.dim
if mesh.comm.rank == 0:
    print(
        f"vertices:      {mesh.topology.index_map(0).size_global:6d} -> "
        f"{periodic_mesh.topology.index_map(0).size_global:6d}"
    )
    print(
        f"geometry nodes:{mesh.geometry.index_map().size_global:6d} -> "
        f"{periodic_mesh.geometry.index_map().size_global:6d}"
    )
    print(
        f"cells:         {mesh.topology.index_map(tdim).size_global:6d} -> "
        f"{periodic_mesh.topology.index_map(tdim).size_global:6d}"
    )

# So the mesh is a torus topologically, while still being drawable as a square. The two
# cells that meet across the seam are genuine neighbours, and a continuous function space
# on the periodic mesh is automatically periodic -- there is no constraint matrix, and no
# boundary condition to apply, because the domain now has no boundary at all.
#
# The other two return values record what happened, for transferring data defined on the
# original mesh. `replaced_vertices` lists the vertices that disappeared, and
# `replacement_map` maps each old (process-local) vertex to its new index;
# {py:func}`scifem.periodic.transfer.transfer_meshtags_to_periodic_mesh` uses them to carry a
# {py:class}`dolfinx.mesh.MeshTags` across.

# ## The variational problem
#
# We solve, on the torus $\Omega$,
#
# $$
# \begin{align}
#   -\Delta u &= f \quad \text{in } \Omega, \\
#   \int_\Omega u \,\mathrm{d}x &= 0.
# \end{align}
# $$
#
# There are no boundary terms, since $\partial\Omega = \emptyset$. The constants are in
# the kernel of the Laplacian, which has two consequences. The solution is only defined
# up to a constant, which the second equation pins; and, because the operator is
# symmetric, the data must be orthogonal to that same kernel,
#
# $$
# \int_\Omega f \,\mathrm{d}x = 0.
# $$
#
# If $f$ violates this the system has **no solution at all**. That failure is quiet: with
# `ksp_type: preonly` PETSc always reports convergence, so `ksp_error_if_not_converged`
# never fires, and MUMPS' null-pivot detection (`mat_mumps_icntl_24`) returns a vector
# regardless. Pinning the mean with a Lagrange multiplier instead, as in the
# {doc}`real_function_space`, makes the system
# nonsingular, so an incompatible $f$ shows up as a wrong answer rather than a plausible
# one.
#
# ### Choosing a solution that actually tests periodicity
#
# We manufacture the problem from an exact solution. The obvious candidates are bad ones:
#
# - $\sin(2\pi x)\sin(2\pi y)$ vanishes identically **on the seam**, so the merged
#   degrees of freedom carry no information and a bug in the identification is invisible.
# - $\cos(2\pi x)\cos(2\pi y)$ has zero normal derivative on all four sides of the unit
#   square, so it *also* solves the homogeneous Neumann problem. An ordinary
#   non-periodic mesh reproduces it exactly as well, and the test proves nothing.
#
# We therefore use
#
# $$
# u_{\text{exact}}(x, y) = \sin(2\pi x) + \sin(2\pi y),
# \qquad f = -\Delta u_{\text{exact}} = 4\pi^2 u_{\text{exact}},
# $$
#
# which is periodic, mean free, $O(1)$ on the seam, and has a normal derivative of
# $\pm 2\pi$ on every side -- so it is emphatically not a homogeneous Neumann solution.


def u_exact(x):
    return ufl.sin(2 * ufl.pi * x[0]) + ufl.sin(2 * ufl.pi * x[1])


# ### Assembling and solving
#
# We use {py:func}`scifem.create_real_functionspace` for the multiplier and solve the
# resulting $2\times 2$ block system.

# +
degree = 2
V = dolfinx.fem.functionspace(periodic_mesh, ("Lagrange", degree))
r_el = basix.ufl.real_element(periodic_mesh.basix_cell(), shape=())
R = dolfinx.fem.functionspace(periodic_mesh, r_el)

W = ufl.MixedFunctionSpace(V, R)
u, lmbda = ufl.TrialFunctions(W)
du, dl = ufl.TestFunctions(W)

x = ufl.SpatialCoordinate(periodic_mesh)
f = -ufl.div(ufl.grad(u_exact(x)))
zero = dolfinx.fem.Constant(periodic_mesh, dolfinx.default_scalar_type(0.0))

a = [
    [ufl.inner(ufl.grad(u), ufl.grad(du)) * ufl.dx, ufl.inner(lmbda, du) * ufl.dx],
    [ufl.inner(u, dl) * ufl.dx, None],
]
L = [ufl.inner(f, du) * ufl.dx, ufl.inner(zero, dl) * ufl.dx]
# -

# Before solving, we check the compatibility condition explicitly. On a mesh with no
# boundary this is the only thing standing between us and an unsolvable system, and it
# costs one reduction.

int_f = assemble_scalar(f * ufl.dx)
assert abs(int_f) < 1e-10, f"source is not mean free: int(f) dx = {int_f}"

problem = dolfinx.fem.petsc.LinearProblem(
    a,
    L,
    kind="mpi",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    },
    petsc_options_prefix="periodic_poisson_",
)
uh, _ = problem.solve()
uh.name = "u"

# ## Verification
#
# ### The solution is mean free
#
# This is what the multiplier enforces, so it is a check on the block system rather than
# on periodicity.

mean = assemble_scalar(uh * ufl.dx)
volume = assemble_scalar(dolfinx.fem.Constant(periodic_mesh, 1.0) * ufl.dx)
if mesh.comm.rank == 0:
    print(f"volume        = {volume:.6f}")
    print(f"int(u) dx     = {mean:.3e}")

# ### The solution is periodic
#
# The mesh has no boundary, so there is nothing to integrate over to check this. Instead
# we evaluate the solution at matching points on the two sides of each seam. Because the
# degrees of freedom there are literally the same, the values must agree to machine
# precision, not merely to discretisation error.

# +
s = np.linspace(0.0, 1.0, 37)[:-1]
left = np.column_stack([np.zeros_like(s), s, np.zeros_like(s)])
right = np.column_stack([np.ones_like(s), s, np.zeros_like(s)])
bottom = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
top = np.column_stack([s, np.ones_like(s), np.zeros_like(s)])


def evaluate(u, points):
    """Evaluate `u` at `points`, on whichever ranks own them."""
    mesh_ = u.function_space.mesh
    tree = dolfinx.geometry.bb_tree(mesh_, mesh_.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh_, candidates, points)
    owned = np.flatnonzero([len(colliding.links(i)) > 0 for i in range(len(points))])
    cells = np.array([colliding.links(i)[0] for i in owned], dtype=np.int32)
    return owned, u.eval(points[owned], cells).reshape(-1)


def seam_jump(u, side_a, side_b):
    ia, va = evaluate(u, side_a)
    ib, vb = evaluate(u, side_b)
    shared = np.intersect1d(ia, ib)
    if len(shared) == 0:
        return 0.0
    diff = np.abs(va[np.searchsorted(ia, shared)] - vb[np.searchsorted(ib, shared)])
    return float(diff.max())


jump_x = mesh.comm.allreduce(seam_jump(uh, left, right), op=MPI.MAX)
jump_y = mesh.comm.allreduce(seam_jump(uh, bottom, top), op=MPI.MAX)
if mesh.comm.rank == 0:
    print(f"max |u(0, y) - u(1, y)| = {jump_x:.3e}")
    print(f"max |u(x, 0) - u(x, 1)| = {jump_y:.3e}")
# -

# ### The solution is *not* a homogeneous Neumann solution
#
# This is the check that distinguishes a working periodic mesh from a broken one. The
# periodic mesh has no boundary to integrate over, so we measure the normal derivative on
# the *original* mesh, after transferring the solution there (the next section explains
# the transfer). For our exact solution
#
# $$
# \int_{\partial\Omega} \left(\frac{\partial u}{\partial n}\right)^2 \mathrm{d}s
#   = 4 \cdot (2\pi)^2 = 16\pi^2 \approx 157.9,
# $$
#
# whereas for $\cos(2\pi x)\cos(2\pi y)$ the same quantity is zero. A solver that
# silently ignored periodicity and applied natural (zero-flux) conditions could not
# produce this field. The computed value is a percent or so off $16\pi^2$, since the
# gradient of a $P_2$ solution is only $P_1$ on the boundary; what matters is that it is
# nowhere near zero.

# ## Visualisation
#
# ### Why the writers get it wrong
#
# `VTXWriter`, and `VTKFile.write_function` for any non-cellwise element, build their
# output point set from the **function space dofmap**: one output point per degree of
# freedom, positioned by pushing the reference interpolation points forward cell by cell.
#
# On a periodic mesh a seam degree of freedom is shared by cells on *opposite sides of
# the domain*, so no single coordinate can represent it -- whichever cell is visited last
# wins. Every cell touching the seam then gets drawn stretched right across the domain.

dof_x = V.tabulate_dof_coordinates()
stretched = 0
for cell in range(periodic_mesh.topology.index_map(tdim).size_local):
    corners = dof_x[V.dofmap.cell_dofs(cell)][:, :2]
    stretched += (corners.max(axis=0) - corners.min(axis=0)).max() > 0.5
stretched = mesh.comm.allreduce(stretched, op=MPI.SUM)
num_cells = periodic_mesh.topology.index_map(tdim).size_global
if mesh.comm.rank == 0:
    print(f"cells VTX would draw stretched across the domain: {stretched}/{num_cells}")

# ### Moving the solution to the parent mesh
#
# The geometry of the periodic mesh still has both sides of the seam, so the fix is to
# put the solution back on the mesh it was built from, where the two sides are distinct
# nodes again. {py:func}`scifem.periodic.create_periodic_mesh` preserves cells -- local cell `c` is the same cell
# in both meshes, with the same geometry dofmap -- so
# {py:func}`scifem.periodic.transfer.transfer_function_to_parent_mesh` is a per-cell
# copy.
#
# ```{admonition} Why a plain per-cell copy is correct
# :class: dropdown
# The dofmaps are not identical at degree $\geq 3$: merging the seam changes the global
# vertex numbering, which flips the order of the two degrees of freedom on an edge in
# `cell_dofs()` for some cells. The copy is still exact, because DOLFINx folds
# permutation-type dof transformations into the dofmap when the space is built. That
# makes `cell_dofs(c)[i]` mean "the dof at the pushforward of reference point $X_i$
# through cell $c$" in each mesh separately -- and since the geometry dofmap is shared,
# that is the same physical point in both. The copy is written in reference-local index
# space, which is exactly the space the permutation has been absorbed into.
#
# This is also its precondition, and
# {py:func}`scifem.periodic.transfer.transfer_function_to_parent_mesh` checks it. An
# element whose transformations are *not* pure permutations, such as `RT`, `N1curl` or
# `BDM`, keeps them out of the dofmap and applies them during assembly, so a per-cell
# copy of one would be wrong; interpolate into Lagrange or discontinuous Lagrange first,
# which the writers require in any case.
# ```


u_parent = transfer_function_to_parent_mesh(uh, mesh)

# The transferred field is only duplicated on the seam. Interpolating into a
# discontinuous space is the other way to make the output well defined, but it gives
# every cell its own copy of every node, which is roughly three times as many points on
# a triangular mesh.

Vdg = dolfinx.fem.functionspace(periodic_mesh, ("Discontinuous Lagrange", degree))
if mesh.comm.rank == 0:
    print(f"output points, periodic space (wrong): {V.dofmap.index_map.size_global:6d}")
    print(
        f"output points, parent mesh:            "
        f"{u_parent.function_space.dofmap.index_map.size_global:6d}"
    )
    print(f"output points, discontinuous space:    {Vdg.dofmap.index_map.size_global:6d}")

# Now that the solution lives on a mesh with a boundary, we can run the Neumann check
# promised above, and compare against the exact solution.

# +
n = ufl.FacetNormal(mesh)
x_parent = ufl.SpatialCoordinate(mesh)
flux = assemble_scalar(ufl.dot(ufl.grad(u_parent), n) ** 2 * ufl.ds)

diff = u_parent - u_exact(x_parent)
error = np.sqrt(assemble_scalar(ufl.inner(diff, diff) * ufl.dx))
if mesh.comm.rank == 0:
    print(
        f"int (du/dn)^2 ds = {flux:.3f}   (16 pi^2 = {16 * np.pi**2:.3f}, "
        "zero for a homogeneous Neumann solution)"
    )
    print(f"L2 error         = {error:.3e}")
# -

# ### Writing and plotting
#
# With the solution on the parent mesh, every writer behaves normally.

# +
with dolfinx.io.VTXWriter(mesh.comm, "periodic_poisson.bp", [u_parent]) as writer:
    writer.write(0.0)

with dolfinx.io.VTKFile(mesh.comm, "periodic_poisson.pvd", "w") as writer:
    writer.write_function(u_parent, 0.0)
# -

# ```{admonition} XDMF is the exception
# :class: tip
# {py:class}`dolfinx.io.XDMFFile` is already correct on the periodic mesh itself, because
# it scatters degrees of freedom onto *geometry* nodes rather than building a point set
# from the dofmap -- and the periodic geometry still has both sides of the seam. It does
# require the function degree to match the mesh degree, so a $P_2$ solution has to be
# interpolated down to $P_1$ first.
# ```

# +
V1 = dolfinx.fem.functionspace(periodic_mesh, ("Lagrange", 1))
u1 = dolfinx.fem.Function(V1, name="u")
u1.interpolate(uh)

with dolfinx.io.XDMFFile(mesh.comm, "periodic_poisson.xdmf", "w") as writer:
    writer.write_mesh(periodic_mesh)
    writer.write_function(u1)
# -

# Finally we plot the transferred solution, warped by its own value. The field runs
# straight off one side of the square and back in on the other, which is what makes the
# mesh periodic; on the un-transferred solution this plot would be unreadable.

# +
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_parent.function_space))
grid.point_data["u"] = u_parent.x.array.real

plotter = pyvista.Plotter()
plotter.add_mesh(
    grid.warp_by_scalar("u", factor=0.15),
    scalars="u",
    cmap="RdBu_r",
    scalar_bar_args={"title": "u"},
)
plotter.add_mesh(grid, style="wireframe", color="black", opacity=0.15)
# Looking down on the square, tilted just enough for the warp to read as height.
plotter.camera_position = [(0.5, -2.0, 1.3), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)]
plotter.camera.zoom(1.05)
if not pyvista.OFF_SCREEN:
    plotter.show()
# -
