# # Poisson on a periodic mesh
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# This example solves the Poisson problem on a doubly periodic unit square built with
# {py:func}`scifem.periodic.create_periodic_mesh`, and covers the four things that
# are easy to get wrong on such a mesh:
#
# 1. **{ref}`Building it <periodic-building>`.** What the indicator and mapping functions
#    do, and how the resulting mesh differs from the original.
# 2. **{ref}`Transferring facet markers <periodic-meshtags>`.** How to move a
#    {py:class}`dolfinx.mesh.MeshTags` from the original mesh to the periodic one.
# 3. **{ref}`Checking the answer <periodic-verification>`.** Why the source has to be mean
#    free, and why the obvious test solutions don't actually distinguish a periodic mesh
#    from a broken one.
# 4. **{ref}`Looking at it <periodic-visualisation>`.** {py:class}`VTXWriter<dolfinx.io.VTXWriter>`
#    and {py:class}`VTKFile<dolfinx.io.VTKFile>` draw a periodic mesh wrongly.

# We start by importing all the relevant libraries we require.

# +
from mpi4py import MPI

import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
from scifem import assemble_scalar
from scifem.periodic import create_periodic_mesh
from scifem.periodic import transfer_function_to_parent_mesh, transfer_meshtags_to_periodic_mesh

# -

# (periodic-building)=
# ## Creating a periodic mesh
#
# We will start from an ordinary DOLFINx mesh, with
# {py:attr}`shared_facet<dolfinx.mesh.GhostMode.shared_facet>` ghost mode,
# This is **required** to ensure that periodicity works correctly across interprocess boundaries.
# {py:func}`scifem.periodic.create_periodic_mesh` checks this and raises if it is missing.
# If you build your mesh by hand, please ensure that you supply
# {py:attr}`dolfinx.mesh.GhostMode.shared_facet` in the mesh construction
# ```{admonition} API compatibility
# :class: tip dropdown
#
# On `main` of DOLFINx, ghost mode is supplied directly to {py:func}`dolfinx.mesh.create_mesh`,
# rather than through the partitioner. Use {py:func}`scifem.compat.create_partitioner` to get
# a partitioner that works on all versions.
# ```

N = 25
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

# Periodicity is described by two functions of a `(3, num_points)` coordinate array.
# The **indicator** marks the vertices that are to disappear, and the **mapping
# function** says, for each marked vertex, which vertex it is identified with. Here we
# remove the $x=1$ and $y=1$ sides and glue them onto $x=0$ and $y=0$.
#
# Both functions are evaluated on the same array, so they must be written to handle the
# corner $(1, 1)$ as well: it is marked once, and the mapping has to shift it in *both*
# directions at once so that it lands on $(0, 0)$.

# + 
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
# -

# The rebuild is purely **topological**; 
# the {py:class}`dolfinx.mesh.Topology` loses a set of vertices,
# because each pair has been merged into a single vertex.
# The **geometry** is untouched: every node of the original
# mesh is still there, with its original coordinates, and the cells
# are the same cells.
# ```{admonition} The new Geometry
# :class: note dropdown
#
# The new {py:class}`dolfinx.mesh.Geometry` is not the same as the original,
# because the new topology might have more cells (local to process) than the
# original, which has to be reflected in the geometry dofmap.
# ```

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

# So the mesh is a torus topologically, while still retaining all its node coordinates.
# The two cells that meet across the seam are genuine neighbours, and a continuous
# function space on the periodic mesh is automatically periodic. 
# There is no [constraint matrix](https://github.com/jorgensd/dolfinx_mpc.git), and no
# boundary condition to apply, because the domain now has no boundary at all.

# + 
def compute_num_exterior_facets(mesh):
    """Count the number of exterior facets on a mesh."""
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    local_exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    return mesh.comm.allreduce(len(local_exterior_facets), op=MPI.SUM)

org_exterior_facets = compute_num_exterior_facets(mesh)
periodic_exterior_facets = compute_num_exterior_facets(periodic_mesh)
if mesh.comm.rank == 0:
    print(f"exterior facets: {org_exterior_facets:6d} -> {periodic_exterior_facets:6d}")
# -

# The other two return values record what happened for transferring data defined on the
# original mesh. `replaced_vertices` lists the vertices that disappeared, and
# `replacement_map` maps each old (process-local) vertex to its new (process-local)
# index; {py:func}`scifem.periodic.transfer_meshtags_to_periodic_mesh` uses them to
# carry a {py:class}`dolfinx.mesh.MeshTags` across.

# (periodic-meshtags)=
# ## Transferring meshtags
#
# In this example we will also observe what happens to a {py:class}`dolfinx.mesh.MeshTags` when
# transferred to a periodic mesh. We start by creating a {py:class}`dolfinx.mesh.MeshTags` object
# on the orignal mesh, marking all facets, including those we want to replace.

interior_marker = -1
top_marker = 1
bottom_marker = 2
left_marker = 3
right_marker = 4
fdim = mesh.topology.dim - 1
org_facet_map = mesh.topology.index_map(fdim)
vec = dolfinx.la.vector(org_facet_map, 1, dtype=np.int32)
facet_values = vec.array
facet_values[:] = interior_marker
facet_values[dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 1.0))] = top_marker
facet_values[dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0))] = bottom_marker
facet_values[dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))] = left_marker
facet_values[dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.0))] = right_marker
vec.scatter_forward()  # locate_entities_boundary only returns local indices. Scatter to update all procs
org_facets = dolfinx.mesh.meshtags(mesh, fdim, np.arange(len(facet_values), dtype=np.int32), facet_values)

# + tags=["hide-input"]
def gather_grid(grid, comm):
    """Merge the per-process grids into one on rank 0. Every other rank gets ``None``.

    Each process holds only its own piece of the mesh, so a plot made without this shows
    one partition rather than the domain.
    """
    pieces = comm.gather(grid, root=0)
    return None if pieces is None else pyvista.merge(pieces)


discrete_viridis = plt.colormaps["viridis"].resampled(2 * N + 1)
facet_mesh = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, fdim, org_facets.indices))
facet_mesh.cell_data["marker"] = org_facets.values
grid = gather_grid(facet_mesh, mesh.comm)
if grid is not None:
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars="marker", cmap=discrete_viridis, show_edges=True,
                     line_width=6, scalar_bar_args={"title": "marker"},
                     clim=(interior_marker, right_marker))
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot("periodic_poisson_facets.png")
# -

# Now we transfer the meshtags to the periodic mesh. The facets on the seam are merged, so
# the new {py:class}`dolfinx.mesh.MeshTags` has fewer facets.

new_facet_tags = transfer_meshtags_to_periodic_mesh(mesh, periodic_mesh, replaced_vertices, org_facets)

# + tags=["hide-input"]
periodic_facet_mesh = pyvista.UnstructuredGrid(
    *dolfinx.plot.vtk_mesh(periodic_mesh, fdim, new_facet_tags.indices)
)
periodic_facet_mesh.cell_data["marker"] = new_facet_tags.values
grid = gather_grid(periodic_facet_mesh, periodic_mesh.comm)
if grid is not None:
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars="marker", cmap=discrete_viridis, show_edges=True,
                     line_width=6, scalar_bar_args={"title": "marker"},
                     clim=(interior_marker, right_marker))
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot("periodic_poisson_periodic_facets.png")
# -

# As we observe, the facets with replacement vertices have been replaced by the facets on the
# opposite side. We measure the area of the facets on the seam and check that it is the
# same as the area of the facets on the opposite side. We can use the `dS` measure on these
# boundaries, as they are now considered interior facets of the periodic mesh.

dS = ufl.Measure("dS", domain=periodic_mesh, subdomain_data=new_facet_tags)
dS_bottom = dS(bottom_marker)
dS_left = dS(left_marker)
area_left = assemble_scalar(dolfinx.fem.Constant(periodic_mesh, 1.0) * dS_left)
area_bottom = assemble_scalar(dolfinx.fem.Constant(periodic_mesh, 1.0) * dS_bottom)
if periodic_mesh.comm.rank == 0:
    print(f"area of facets on left side: {area_left:.6f}", flush=True)
    print(f"area of facets on bottom side: {area_bottom:.6f}", flush=True)
assert np.isclose(area_left, 1.0)
assert np.isclose(area_bottom, 1.0)

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
# If $f$ violates this, the problem above has no solution at all.
#
# ```{admonition} Compatibility condition with Lagrange multipliers
# :name: compatibility-multiplier
# :class: dropdown
#
# We do not enforce the mean with a boundary condition, but with a Lagrange multiplier
# $\lambda$, as in {doc}`real_function_space`, which makes the discrete system
# nonsingular. That is worth being precise about, because the multiplier does not merely
# pin the constant -- it also *absorbs* any violation of the compatibility condition. The
# system actually solved is
#
# $$
# \begin{align}
#   -\Delta u + \lambda &= f \quad \text{in } \Omega, \\
#   \int_\Omega u \,\mathrm{d}x &= 0,
# \end{align}
# $$
#
# and testing the first equation against $v = 1$ gives $\lambda\,|\Omega| = \int_\Omega
# f \,\mathrm{d}x$: the multiplier comes out as the mean of the source,
# $\bar f := |\Omega|^{-1}\int_\Omega f \,\mathrm{d}x$. So the system is
# nonsingular for *any* $f$, and the solver will not complain. What it returns is the
# solution of $-\Delta u = f - \bar f$, a different problem from the one we posed
# whenever $\bar f \neq 0$.
# ```
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
# which is periodic, mean free, non-zero on the seam, and has a normal derivative of
# $\pm 2\pi$ on every side. This implies that it is not a solution to the homogeneous
# Neumann problem.


def u_exact(x):
    return ufl.sin(2 * ufl.pi * x[0]) + ufl.sin(2 * ufl.pi * x[1])


# ### Assembling and solving
#
# The multiplier lives in a "real" space: one degree of freedom for the whole domain. We
# build it from {py:func}`basix.ufl.real_element` and solve the resulting $2\times 2$
# block system.

# +
degree = 2
V = dolfinx.fem.functionspace(periodic_mesh, ("Lagrange", degree))
r_el = basix.ufl.real_element(periodic_mesh.basix_cell(), value_shape=())
R = dolfinx.fem.functionspace(periodic_mesh, r_el)

W = ufl.MixedFunctionSpace(V, R)
u, lmbda = ufl.TrialFunctions(W)
du, dl = ufl.TestFunctions(W)

x = ufl.SpatialCoordinate(periodic_mesh)
f = -ufl.div(ufl.grad(u_exact(x)))

# We use {py:class}`ufl.ZeroBaseForm` to make the RHS block for the {py:class}`LinearProblem<dolfinx.fem.petsc.LinearProblem>`
# constructor, which expects a list of forms for the RHS.
a = ufl.inner(ufl.grad(u), ufl.grad(du)) * ufl.dx + ufl.inner(lmbda, du) * ufl.dx + ufl.inner(u, dl) * ufl.dx
L = [ufl.inner(f, du) * ufl.dx, ufl.ZeroBaseForm((dl,))]
# -

# The compatibility condition is checked here rather than relied on. As
# {ref}`the admonition above <compatibility-multiplier>` sets out, a source that violates
# it is not caught by the solve -- it silently changes the equation being solved -- and one
# reduction is enough to rule that out. After the fact $\lambda$ carries the same
# information: a non-zero multiplier in the solution is exactly the mean that was absorbed.

# +
int_f = assemble_scalar(f * ufl.dx)
assert abs(int_f) < 1e-10, f"source is not mean free: int(f) dx = {int_f}"

problem = dolfinx.fem.petsc.LinearProblem(
    ufl.extract_blocks(a),
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
# -

# (periodic-verification)=
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
# The mesh has no boundary, so there is nothing to integrate over. The seam, however, is
# now made of *interior* facets carrying the markers transferred above, so `dS` reaches
# it: the continuity of the space across the seam is one integral over the whole seam at
# once, rather than a comparison at sampled points.
#
# ```{admonition} What the vanishing jump does and does not prove
# :class: dropdown
#
# In a continuous space the jump across an interior facet is zero by construction: the two
# sides read the same degrees of freedom. So the integral vanishes for *every* function in
# `V`, not only for this solution.
#
# What it confirms is that the seam facets really did become interior facets, and that the
# degrees of freedom on them were merged with a consistent orientation. It cannot tell a
# correct pairing from a wrong one, as a seam glued with a shift would pass just as cleanly.
#
# If $V$ were a discontinuous space the integral would no longer vanish identically, since
# nothing there forces the two sides of a facet to agree. A seam jump out of proportion to
# the jumps on ordinary interior facets would then show that the coupling terms of the DG
# scheme are not reaching across the seam.
# ```

seam = dS(bottom_marker) + dS(left_marker)
seam_jump = assemble_scalar(ufl.jump(uh) ** 2 * seam)
if mesh.comm.rank == 0:
    print(f"int jump(u)^2 dS = {seam_jump:.3e}")

# ### The solution is *not* a homogeneous Neumann solution
#
# This is the check that distinguishes a working periodic mesh from a broken one, and the
# same measure makes it. The normal derivative on the seam is exactly what a homogeneous
# Neumann solution is not allowed to have. The `bottom` and `left` markers carry the whole
# seam, of total length $2$, and $\partial u/\partial n = \pm 2\pi$ along both, so
#
# $$
# \int_\Gamma \left(\frac{\partial u}{\partial n}\right)^2 \mathrm{d}s
#   = 2 \cdot (2\pi)^2 = 8\pi^2 \approx 79.0,
# $$
#
# whereas for $\cos(2\pi x)\cos(2\pi y)$ the same quantity is zero. A solver that
# silently ignored periodicity and applied natural (zero-flux) conditions could not
# produce this field.
#
# The gradient of a $P_2$ function is discontinuous across a facet, so we should consider
# the values from both sides. The `"+"` and `"-"` restrictions in DOLFINx are arbitrary
# unless the integration entities are oriented manually (see
# {py:func}`scifem.compute_interface_data` or
# [Consistent orientations](https://scientificcomputing.github.io/fenics-in-the-wild/src/ucs/emi/emi_primal_single.html#consistent-restrictions)).
# We therefore compute the average of the two sides, which is independent of the orientation.

n_periodic = ufl.FacetNormal(periodic_mesh)
flux = assemble_scalar(ufl.avg(ufl.dot(ufl.grad(uh), n_periodic) ** 2) * seam)
if mesh.comm.rank == 0:
    print(
        f"int (du/dn)^2 dS = {flux:.3f}   (8 pi^2 = {8 * np.pi**2:.3f}, "
        "zero for a homogeneous Neumann solution)"
    )

# (periodic-visualisation)=
# ## Visualisation
#
# ### Why the writers get it wrong
#
# {py:class}`VTXWriter<dolfinx.io.VTXWriter>`, and
# {py:meth}`VTKFile.write_function<dolfinx.io.VTKFile.write_function>` for
# any non-cellwise element, build their output point set from the **function space
# dofmap**: one output point per degree of freedom, positioned by pushing the reference
# interpolation points forward cell by cell.
#
# On a periodic mesh a seam degree of freedom is shared by cells on *opposite sides of
# the domain*, so no single coordinate can represent it, whichever cell is visited last
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
# put the solution back on the mesh it was built from.
# {py:func}`scifem.periodic.create_periodic_mesh` preserves cells, so the local
# cell `c` is the same cell in both meshes, with the same geometry dofmap (up to extra
# ghost cells in the new periodic mesh).
#
# ```{admonition} Why the cell-wise transfer is correct
# :class: dropdown
#
# Merging the seam can change a cell's orientation, so the two meshes need not agree on how
# a cell's degrees of freedom are transformed. For a space whose transformations are
# permutations, DOLFINx permutes the dofmap at construction, so a given reference-local
# index already names the same physical point in both meshes.
#
# Elements that instead apply their transformations at assembly, such as `RT`, `N1curl` or
# `BDM`, are handled too: the cell-wise interpolation that
# {py:func}`scifem.periodic.transfer_function_to_parent_mesh` performs accounts for the
# orientations the two meshes disagree on. The writers still require Lagrange or
# discontinuous Lagrange, so interpolate before writing.
# ```


u_parent = transfer_function_to_parent_mesh(uh, mesh)

# The transferred field is only duplicated on the seam. Interpolating into a
# discontinuous space on the perioidc mesh is the other way to make the output well
# defined, but it gives every cell its own copy of every node, which is roughly three
# times as many points on a triangular mesh.

Vdg = dolfinx.fem.functionspace(periodic_mesh, ("Discontinuous Lagrange", degree))
if mesh.comm.rank == 0:
    print(f"output points, periodic space (wrong): {V.dofmap.index_map.size_global:6d}")
    print(
        f"output points, parent mesh:            "
        f"{u_parent.function_space.dofmap.index_map.size_global:6d}"
    )
    print(f"output points, discontinuous space:    {Vdg.dofmap.index_map.size_global:6d}")

# The solve happened on the periodic mesh and the output lives on the parent mesh, but the
# two carry the same cells with the same coordinates, so the same error integral can be
# formed on either. Forming it on both is a check on the transfer itself: a per-cell copy
# that placed any degree of freedom wrongly would not reproduce the number.

# +
diff_periodic = uh - u_exact(x)
error_periodic = np.sqrt(assemble_scalar(ufl.inner(diff_periodic, diff_periodic) * ufl.dx))

x_parent = ufl.SpatialCoordinate(mesh)
diff_parent = u_parent - u_exact(x_parent)
error_parent = np.sqrt(assemble_scalar(ufl.inner(diff_parent, diff_parent) * ufl.dx))

if mesh.comm.rank == 0:
    print(f"L2 error, periodic mesh = {error_periodic:.3e}")
    print(f"L2 error, parent mesh   = {error_parent:.3e}")
assert np.isclose(error_periodic, error_parent)
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
# :class: tip dropdown
# {py:class}`dolfinx.io.XDMFFile` is already correct on the periodic mesh itself, because
# it scatters degrees of freedom onto *geometry* nodes rather than building a point set
# from the dofmap, and the periodic geometry still has both sides of the seam. It does
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

# + tags =["hide-input"]
solution_mesh = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_parent.function_space))
solution_mesh.point_data["u"] = u_parent.x.array.real

grid = gather_grid(solution_mesh, mesh.comm)
if grid is not None:
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
    else:
        plotter.screenshot("periodic_poisson_solution.png")
# -
