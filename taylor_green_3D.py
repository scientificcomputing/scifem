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
    def __init__(self, nu, V0, L):
        self.nu = nu
        self.V0 = V0
        self.L = L

    def eval_x(self, x: npt.NDArray[np.float64]) -> npt.NDArray[dolfinx.default_scalar_type]:
        return self.V0 * np.sin(x[0] / self.L) * np.cos(x[1] / self.L) * np.sin(x[2] / self.L)

    def eval_y(self, x: npt.NDArray[np.float64]) -> npt.NDArray[dolfinx.default_scalar_type]:
        return -self.V0 * np.cos(x[0] / self.L) * np.sin(x[1] / self.L) * np.cos(x[2] / self.L)

    def eval_z(self, x: npt.NDArray[np.float64]) -> npt.NDArray[dolfinx.default_scalar_type]:
        return np.zeros_like(x[0])


parser = argparse.ArgumentParser(description="Taylor-Green 3D demo")
parser.add_argument(
    "-N",
    "--refinement",
    type=int,
    dest="N",
    required=True,
    help="Number of elements in each direction.",
)
parser.add_argument("-u", dest="u_deg", type=int, help="Degree of velocity space", default=2)
parser.add_argument("-p", dest="p_deg", type=int, help="Degree of pressure space", default=1)
parser.add_argument("-L", dest="L", type=float, help="Length of domain", default=1)
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
parser.add_argument("--save-frequency", dest="save_frequency", type=int, default=10)
inputs = parser.parse_args()

logger = logging.getLogger("oasisx")
logger.setLevel(getattr(logging, inputs.log_level.upper(), logging.INFO))

V0 = 1.0
L = inputs.L
t_c = L / V0
T_end = 20 * t_c
RE = 1600
nu = 1 / (RE * (V0 * L))
dt = 0.001

num_steps = int((T_end) // dt)

assert inputs.u_deg > inputs.p_deg
el_u = ("Lagrange", inputs.u_deg)
el_p = ("Lagrange", inputs.p_deg)
f = None
options = {"low_memory_version": inputs.lm}

solver_options = {
    "tentative": {
        "ksp_type": "bcgs",
        "pc_type": "jacobi",
    },
    "pressure": {
        "ksp_type": "minres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
    },
    "scalar": {
        "ksp_type": "cg",
        "pc_type": "sor",
    },
}

N = inputs.N
logger.info(f"Creating mesh with {N} elements in each direction.")
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,
    [[0, 0, 0], [2 * np.pi * L, 2 * np.pi * L, 2 * np.pi * L]],
    [inputs.N, inputs.N, inputs.N],
    cell_type=dolfinx.mesh.CellType.tetrahedron,
)
# Convert mesh to periodic mesh
L_min = [mesh.comm.allreduce(np.min(mesh.geometry.x[:, i]), op=MPI.MIN) for i in range(3)]
L_max = [mesh.comm.allreduce(np.max(mesh.geometry.x[:, i]), op=MPI.MAX) for i in range(3)]


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
mesh, replaced_vertices, replacement_map = create_periodic_mesh(mesh, indicator, mapping)

mesh.topology.create_entities(mesh.topology.dim - 1)

dim = mesh.topology.dim - 1
# Locate facets for boundary conditions and create meshtags
mesh.topology.create_connectivity(dim, dim + 1)
logger.info("Mesh created and connectivity established.")


u_ex = U(nu=nu, V0=V0, L=L)
logger.info("Setting up boundary conditions.")

bcs_u: List[List[oasisx.DirichletBC]] = [[] for _ in range(mesh.geometry.dim)]
bcs_p: List[oasisx.PressureBC] = []

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
solver._u2[0].interpolate(u_ex.eval_x)
solver._u2[1].interpolate(u_ex.eval_y)
solver._u2[2].interpolate(u_ex.eval_z)
solver._u1[0].interpolate(u_ex.eval_x)
solver._u1[1].interpolate(u_ex.eval_y)
solver._u1[2].interpolate(u_ex.eval_z)

# Set initial conditions for pressure
x = ufl.SpatialCoordinate(mesh)
man_p = (1 / 16) * (ufl.cos(2 * x[0]) + ufl.cos(2 * x[1])) * (ufl.cos(2 * x[2]) + 2)

p_expr = dolfinx.fem.Expression(man_p, solver._Q.element.interpolation_points())
solver._p.interpolate(p_expr)
logger.info("Initial conditions set.")


V_out = dolfinx.fem.functionspace(mesh, ("DG", inputs.u_deg, (mesh.geometry.dim,)))
v_out = dolfinx.fem.Function(V_out)

vtxu = dolfinx.io.VTXWriter(
    mesh.comm,
    "u.bp",
    [v_out],
    engine="BP5",
    mesh_policy=dolfinx.cpp.io.VTXMeshPolicy.reuse,
)


vtxp = dolfinx.io.VTXWriter(
    mesh.comm,
    "p.bp",
    [solver._p],
    engine="BP5",
    mesh_policy=dolfinx.cpp.io.VTXMeshPolicy.reuse,
)
t = 0
save_interval = inputs.save_frequency
for i in range(num_steps):
    print(f"{i}/{num_steps}", end="\r")
    t += float(dt)
    logger.debug(f"Time step {i + 1}/{num_steps}, solving at t={t:.3f}.")
    solver.solve(dt, nu, max_iter=1)
    v_out.interpolate(solver.u)
    if i % save_interval == 0:
        vtxu.write(t)
        vtxp.write(t)

vtxu.close()
vtxp.close()
logger.info("Simulation completed. Output saved.")
