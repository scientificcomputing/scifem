# # Real function spaces
# In this example, we create a real function space on a unit square mesh.

from mpi4py import MPI
import numpy as np
import dolfinx
from scifem import create_real_functionspace

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 10, 10, dolfinx.mesh.CellType.triangle, dtype=np.float64
)
V = create_real_functionspace(mesh)
print(V)
