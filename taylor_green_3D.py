import argparse
import logging
from typing import List
from script import create_periodic_mesh
from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl

import oasisx


class U:
    def __init__(self, t, nu):
        self.t = t
        self.nu = nu

    def eval_x(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[dolfinx.default_scalar_type]:
        return (
            -np.cos(np.pi * x[0])
            * np.sin(np.pi * x[1])
            * np.cos(np.pi * x[2])
            * np.exp(-3.0 * self.nu * np.pi**2 * float(self.t))
        )

    def eval_y(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[dolfinx.default_scalar_type]:
        return (
            np.cos(np.pi * x[1])
            * np.sin(np.pi * x[0])
            * np.cos(np.pi * x[2])
            * np.exp(-3.0 * self.nu * np.pi**2 * float(self.t))
        )

    def eval_z(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[dolfinx.default_scalar_type]:
        return (
            -2
            * np.cos(np.pi * x[2])
            * np.sin(np.pi * x[0])
            * np.sin(np.pi * x[1])
            * np.exp(-3.0 * self.nu * np.pi**2 * float(self.t))
        )


parser = argparse.ArgumentParser(description="Taylor-Green 3D demo")
parser.add_argument(
    "-N",
    "--refinement",
    type=int,
    dest="N",
    required=True,
    help="Number of elements in each direction.",
)
parser.add_argument(
    "-T0",
    "--T-start",
    dest="T_start",
    type=float,
    help="Start time of simulation",
    default=0,
)
parser.add_argument(
    "-T1", "--T-end", dest="T_end", type=float, help="End time of simulation", default=1
)
parser.add_argument("-dt", dest="dt", type=float, help="Time step", default=0.1)
parser.add_argument(
    "-nu", dest="nu", type=float, help="Kinematic viscosity", default=0.01
)
parser.add_argument(
    "-u", dest="u_deg", type=int, help="Degree of velocity space", default=2
)
parser.add_argument(
    "-p", dest="p_deg", type=int, help="Degree of pressure space", default=1
)
parser.add_argument(
    "-lm",
    "--low-memory",
    dest="lm",
    action="store_true",
    help="Use low memory version of Oasisx",
    default=False,
)
parser.add_argument(
    "-r",
    "--rotational",
    dest="rot",
    action="store_true",
    help="Use rotational formulation of pressure update",
    default=False,
)
parser.add_argument(
    "-log",
    "--log-level",
    dest="log_level",
    type=str,
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level",
    default="INFO",
)
inputs = parser.parse_args()

logger = logging.getLogger("oasisx")
logger.setLevel(getattr(logging, inputs.log_level.upper(), logging.INFO))

assert inputs.T_start < inputs.T_end
T_start, T_end, dt, nu = inputs.T_start, inputs.T_end, inputs.dt, inputs.nu
num_steps = int((T_end - T_start) // dt)

assert inputs.u_deg > inputs.p_deg
el_u = ("Lagrange", inputs.u_deg)
el_p = ("DPC", inputs.p_deg)
f = None
options = {"low_memory_version": inputs.lm, "gamma":1000}

solver_options = {
    "tentative": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    "pressure": {"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type": "mumps"},
    "scalar": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
}

N = inputs.N
logger.info(f"Creating mesh with {N} elements in each direction.")
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [[-1, -1, -1], [1, 1, 1]],
    [inputs.N, inputs.N, inputs.N],
    cell_type=dolfinx.mesh.CellType.hexahedron,
)
# Convert mesh to periodic mesh
L_min = [
    mesh.comm.allreduce(np.min(mesh.geometry.x[:, i]), op=MPI.MIN) for i in range(3)
]
L_max = [
    mesh.comm.allreduce(np.max(mesh.geometry.x[:, i]), op=MPI.MAX) for i in range(3)
]


def i_x(x):
    return np.isclose(x[0], L_min[0])


def i_y(x):
    return np.isclose(x[1], L_min[1])


def i_z(x):
    return np.isclose(x[2], L_min[2])


def indicator(x):
    return i_x(x) | i_y(x) | i_z(x)


def mapping(x):
    values = x.copy()
    values[0] += i_x(x) * (L_max[0] - L_min[0])
    values[1] += i_y(x) * (L_max[1] - L_min[1])
    values[2] += i_z(x) * (L_max[2] - L_min[2])
    return values


mesh.topology.create_entities(mesh.topology.dim - 1)
print(f"NUm vertices pre refinement {mesh.topology.index_map(0).size_global}")
print(f"NUm facets pre refinement {mesh.topology.index_map(2).size_global}")
print(f"NUm cells pre refinement {mesh.topology.index_map(3).size_global}")

print(MPI.COMM_WORLD.rank, f"map x from {L_min} to {L_max}")
mesh, replaced_vertices, replacement_map = create_periodic_mesh(
    mesh, indicator, mapping
)

mesh.topology.create_entities(mesh.topology.dim - 1)
print(f"NUm vertices post refinement {mesh.topology.index_map(0).size_global}")
print(f"NUm facets post refinement {mesh.topology.index_map(2).size_global}")
print(f"NUm cells post refinement {mesh.topology.index_map(3).size_global}")
with dolfinx.io.XDMFFile(mesh.comm, "Periodic_mesh.xdmf", "w") as xdmf: 
    xdmf.write_mesh(mesh)

dim = mesh.topology.dim - 1
# Locate facets for boundary conditions and create meshtags
mesh.topology.create_connectivity(dim, dim + 1)
logger.info("Mesh created and connectivity established.")

facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
values = np.full_like(facets, 3, dtype=np.int32)
sort = np.argsort(facets)
facet_tags = dolfinx.mesh.meshtags(mesh, dim, facets[sort], values[sort])
logger.info("Boundary facets located and tagged.")

u_time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(T_start))
p_time = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(T_start - dt / 2.0))
u_ex = U(t=u_time, nu=nu)
logger.info("Setting up boundary conditions.")
# bcs_u = [
#     [
#         oasisx.DirichletBC(
#             u_ex.eval_x, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, 3)
#         )
#     ],
#     [
#         oasisx.DirichletBC(
#             u_ex.eval_y, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, 3)
#         )
#     ],
#     [
#         oasisx.DirichletBC(
#             u_ex.eval_z, oasisx.LocatorMethod.TOPOLOGICAL, (facet_tags, 3)
#         )
#     ],
# ]
bcs_u: List[List[oasisx.DirichletBC]] = None
bcs_p: List[oasisx.PressureBC] = None

# Initialize the fractional step solver with the AB-CN scheme.
solver = oasisx.FractionalStep_AB_CN(
    mesh,
    el_u,
    el_p,
    bcs_u=bcs_u,
    bcs_p=bcs_p,
    rotational=inputs.rot,
    solver_options=solver_options,
    options=options,
    body_force=f,
)
logger.info("Solver initialized.")

# Set initial conditions for velocity
logger.info("Setting up initial conditions.")
u_time.value = T_start - dt
solver._u2[0].interpolate(u_ex.eval_x)
solver._u2[1].interpolate(u_ex.eval_y)
solver._u2[2].interpolate(u_ex.eval_z)
u_time.value = T_start
solver._u1[0].interpolate(u_ex.eval_x)
solver._u1[1].interpolate(u_ex.eval_y)
solver._u1[2].interpolate(u_ex.eval_z)

# Set initial conditions for pressure
x = ufl.SpatialCoordinate(mesh)
man_p = (
    -0.25
    * (
        ufl.cos(2 * ufl.pi * x[0])
        + ufl.cos(2 * ufl.pi * x[1])
        + ufl.cos(2 * ufl.pi * x[2])
    )
    * ufl.exp(-6 * ufl.pi**2 * nu * p_time)
)
p_expr = dolfinx.fem.Expression(man_p, solver._Q.element.interpolation_points)
solver._p.interpolate(p_expr)
logger.info("Initial conditions set.")


V_out = dolfinx.fem.functionspace(mesh, ("DG", inputs.u_deg, (mesh.geometry.dim, )))
v_out = dolfinx.fem.Function(V_out)
vtxu = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [v_out], engine="BP5")


Q_out = dolfinx.fem.functionspace(mesh, ("DG", inputs.u_deg))
q_out = dolfinx.fem.Function(Q_out)
vtxp = dolfinx.io.VTXWriter(mesh.comm, "p.bp", [q_out], engine="BP5")
for i in range(num_steps):
    u_time.value += dt
    p_time.value += dt
    print(float(u_time.value))
    logger.debug(f"Time step {i + 1}/{num_steps}, solving at t={u_time.value:.3f}.")
    solver.solve(dt, nu, max_iter=1)
    v_out.interpolate(solver.u)
    vtxu.write(u_time.value)
    q_out.interpolate(solver._p)
    vtxp.write(p_time.value)

vtxu.close()
vtxp.close()
logger.info("Simulation completed. Output saved.")
