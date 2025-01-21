from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.io import VTXWriter
import ufl
import gmsh
import dolfinx
from petsc4py import PETSc
from basix.ufl import element
from mpi4py import MPI
import time
from script import create_periodic_mesh, transfer_meshtags_to_periodic_mesh
import numpy as np

comm = MPI.COMM_WORLD

# Register starting time
MPI.WTIME_IS_GLOBAL = True
start_time = MPI.Wtime()


# inpurt parameters
Lx = 0.25
Ly = 0.15
Lz = 0.5
wl = 0.3
theta = 30 * np.pi / 180
TE = True

# Define region tags
air_tag = 1
top_tag = 2
bottom_tag = 3
left_tag = 4
right_tag = 5
front_tag = 6
back_tag = 7
pec_tag = (front_tag, back_tag)

# Generate mesh
model = None
gmsh.initialize()

if comm.rank == 0:
    gmsh.model.add("geometry")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.04)
    gmsh.model.occ.addBox(-Lx / 2, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], tag=1, name="air")

    eps = 0.001

    min_z = 0
    max_z = Lz
    dimtags_d = gmsh.model.getEntitiesInBoundingBox(
        -Lx / 2 - eps, -eps, min_z - eps, Lx + eps, +eps, max_z + eps, 2
    )
    gmsh.model.addPhysicalGroup(
        2, list(zip(*dimtags_d))[1], tag=front_tag, name="front"
    )
    dimtags_l = gmsh.model.getEntitiesInBoundingBox(
        -Lx / 2 - eps,
        -Ly / 2 - eps,
        min_z - eps,
        -Lx / 2 + eps,
        Ly + eps,
        max_z + eps,
        2,
    )
    gmsh.model.addPhysicalGroup(2, list(zip(*dimtags_l))[1], tag=left_tag, name="left")

    dimtags_u = gmsh.model.getEntitiesInBoundingBox(
        -Lx / 2 - eps, Ly - eps, min_z - eps, Lx + eps, Ly + eps, max_z + eps, 2
    )
    gmsh.model.addPhysicalGroup(2, list(zip(*dimtags_u))[1], tag=back_tag, name="back")
    dimtags_r = gmsh.model.getEntitiesInBoundingBox(
        Lx / 2 - eps, -eps, min_z - eps, Lx / 2 + eps, Ly + eps, max_z + eps, 2
    )
    gmsh.model.addPhysicalGroup(
        2, list(zip(*dimtags_r))[1], tag=right_tag, name="right"
    )

    dimtags = gmsh.model.getEntitiesInBoundingBox(
        -Lx / 2 - eps, -eps, max_z - eps, Lx + eps, Ly + eps, max_z + eps, 2
    )
    gmsh.model.addPhysicalGroup(2, list(zip(*dimtags))[1], tag=top_tag, name="top")
    dimtags = gmsh.model.getEntitiesInBoundingBox(
        -Lx / 2 - eps, -eps, min_z - eps, Lx + eps, Ly + eps, min_z + eps, 2
    )
    gmsh.model.addPhysicalGroup(
        2, list(zip(*dimtags))[1], tag=bottom_tag, name="bottom"
    )

    translation = [
        1,
        0,
        0,
        Lx,  # X-axis translation component
        0,
        1,
        0,
        0,  # Y-axis translation component
        0,
        0,
        1,
        0,  # Z-axis translation component
        0,
        0,
        0,
        1,
    ]
    gmsh.model.mesh.setPeriodic(
        2, list(zip(*dimtags_r))[1], list(zip(*dimtags_l))[1], translation
    )

    translation = [
        1,
        0,
        0,
        0,  # X-axis translation component
        0,
        1,
        0,
        Ly,  # Y-axis translation component
        0,
        0,
        1,
        0,  # Z-axis translation component
        0,
        0,
        0,
        1,
    ]
    gmsh.model.mesh.setPeriodic(
        2, list(zip(*dimtags_u))[1], list(zip(*dimtags_d))[1], translation
    )
    gmsh.model.mesh.generate(3)

    gmsh.write("mesh.msh")

model = comm.bcast(model, root=0)
partitioner = dolfinx.cpp.mesh.create_cell_partitioner(
    dolfinx.mesh.GhostMode.shared_facet
)
model = model_to_mesh(gmsh.model, comm, 0, gdim=3, partitioner=partitioner)
gmsh.finalize()
domain1 = model.mesh
cell_tags1 = model.cell_tags
facet_tags1 = model.facet_tags
# Convert mesh to periodic mesh
L_min = comm.allreduce(np.min(domain1.geometry.x[:, 0]), op=MPI.MIN)
L_max = comm.allreduce(np.max(domain1.geometry.x[:, 0]), op=MPI.MAX)
Ly_min = comm.allreduce(np.min(domain1.geometry.x[:, 1]), op=MPI.MIN)
Ly_max = comm.allreduce(np.max(domain1.geometry.x[:, 1]), op=MPI.MAX)


def i_x(x):
    return np.isclose(x[0], L_min)


def i_y(x):
    return np.isclose(x[1], Ly_min)


def indicator(x):
    return i_x(x) | i_y(x)


def mapping(x):
    values = x.copy()
    values[0] += i_x(x) * (L_max - L_min)
    values[1] += i_y(x) * (Ly_max - Ly_min)
    return values


print(MPI.COMM_WORLD.rank, f"map x from {L_min} to {L_max}")
domain2, replaced_vertices, replacement_map = create_periodic_mesh(
    domain1, indicator, mapping
)
facet_tags2 = transfer_meshtags_to_periodic_mesh(
    domain1, domain2, replaced_vertices, facet_tags1
)
cell_tags2 = transfer_meshtags_to_periodic_mesh(
    domain1, domain2, replaced_vertices, cell_tags1
)
domain2.topology.create_connectivity(domain2.topology.dim - 1, domain2.topology.dim)


# def indicator(x):
#     return np.isclose(x[1], Ly_min)


# def mapping(x):
#     values = x.copy()
#     values[1] += Ly_max - Ly_min
#     return values


# print(MPI.COMM_WORLD.rank, f"map y from {Ly_min} to {Ly_max}")
# domain, replaced_vertices, replacement_map = create_periodic_mesh(
#     domain2, indicator, mapping
# )
# exit()

# facet_tags = transfer_meshtags_to_periodic_mesh(
#     domain2, domain, replaced_vertices, facet_tags2
# )
# cell_tags = transfer_meshtags_to_periodic_mesh(
#     domain2, domain, replaced_vertices, cell_tags2
# )

cell_tags = cell_tags2
domain = domain2
facet_tags = facet_tags2

domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
import dolfinx

with dolfinx.io.XDMFFile(comm, "mesh.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tags, domain.geometry)

if comm.rank == 0:
    print("Info    : Create periodic mesh.")


# Define elements, spaces and measures
curl_el = element("N1curl", domain.basix_cell(), 2)
V = fem.functionspace(domain, curl_el)

dx = ufl.Measure(
    "dx",
    domain,
    subdomain_data=cell_tags,  # , metadata={"quadrature_degree": 4}
)
ds = ufl.Measure(
    "exterior_facet",
    domain,
    subdomain_data=facet_tags,
    metadata={"quadrature_degree": 4},
)

U = ufl.TrialFunction(V)
testU = ufl.TestFunction(V)
num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
print(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
# 24146
# assert num_dofs_global == 24146

# Define problem constants and variables
k0 = fem.Constant(domain, PETSc.ScalarType(2 * np.pi / wl))
jkx = fem.Constant(domain, PETSc.ScalarType(1j * k0.value * np.sin(theta)))
jky = fem.Constant(domain, PETSc.ScalarType(-1j * k0.value * np.cos(theta)))
x, y, z = ufl.SpatialCoordinate(domain)
TE = False
if TE:
    Uinc = ufl.as_vector((0, 1, 0))
else:
    Uinc = ufl.as_vector((ufl.cos(theta), 0, 100 * ufl.sin(theta)))
n = ufl.FacetNormal(domain)
ki = ufl.as_vector((np.sin(theta), 0, -np.cos(theta)))
kr = ufl.as_vector((np.sin(theta), 0, np.cos(theta)))

# Define problem
F = (
    -ufl.inner(
        ufl.curl(U) + ufl.cross(ufl.as_vector((jkx, 0, 0)), U),
        ufl.curl(testU) + ufl.cross(ufl.as_vector((jkx, 0, 0)), testU),
    )
    * dx
    + k0**2 * ufl.inner(U, testU) * dx
    + 1j * k0 * ufl.inner(ufl.cross(U, ki), ufl.cross(testU, n)) * ds(top_tag)
    + 1j
    * k0
    * ufl.inner(ufl.cross(Uinc, kr - ki), ufl.cross(testU, n))
    * ds(bottom_tag)
    + 1j * k0 * ufl.inner(ufl.cross(U, kr), ufl.cross(testU, n)) * ds(bottom_tag)
)

a, L = ufl.lhs(F), ufl.rhs(F)

# PEC Boundary conditions
bcs = []
zero = fem.Function(V)
zero.x.array[:] = 0

# if TE:
#     for i in pec_tag:
#         pec_dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facet_tags.find(i))
#         bc=fem.dirichletbc(zero, pec_dofs)
#         bcs.append(bc)


# Solve problem
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)  #
U = problem.solve()

# save solution
W = fem.functionspace(domain, ("Discontinuous Lagrange", 2, (3,)))
E_dg = fem.Function(W)
E_dg.interpolate(fem.Expression(U * ufl.exp(jkx * x), W.element.interpolation_points))

with VTXWriter(domain.comm, "E3d.bp", E_dg) as f:
    f.write(0.0)
if comm.rank == 0:
    print("Info    : Saved solution.")

if comm.rank == 0:
    print(
        "Info    : Done! Time: "
        + time.strftime("%H:%M:%S", time.gmtime(MPI.Wtime() - start_time))
    )
