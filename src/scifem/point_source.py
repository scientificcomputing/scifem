# Create a point source for Poisson problem
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT

from __future__ import annotations

from packaging.version import Version
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import ufl

from .utils import unroll_dofmap

__all__ = ["PointSource"]


class PointSource:
    """Class for defining a point source in a given function space."""

    def __init__(
        self,
        V: dolfinx.fem.FunctionSpace,
        points: npt.NDArray[np.float32] | npt.NDArray[np.float64],
        magnitude: np.floating | np.complexfloating = dolfinx.default_scalar_type(1),
    ) -> None:
        """Initialize a point source.

        Args:
            V: The function space the point source is defined in.
            points: The points where the point source is located.
                Input shape: ``(num_points, 3)``
            magnitude: The magnitudes of the point sources.

        Note:
            Points should only be defined on one process. If they are sent in
            from multiple processes, multiple point sources will be created.

        Note:
            If the point source is outside the mesh, a ``ValueError`` will be raised.
        """
        self._function_space = V
        if V.dofmap.bs > 1 and dolfinx.__version__ == "0.8.0":
            raise NotImplementedError(
                "Block function spaces are not supported in dolfinx 0.8.0. "
                "Please upgrade dolfinx"
            )
        self._input_points = points
        self._magnitude = magnitude
        # Initialize empty arrays
        self._points = np.empty((0, 3), dtype=points.dtype)
        self._cells = np.empty(0, dtype=np.int32)
        num_dofs = self._function_space.dofmap.dof_layout.num_dofs * self._function_space.dofmap.bs
        self._basis_values = np.empty(
            (0, num_dofs), dtype=self._function_space.mesh.geometry.x.dtype
        )

        self.recompute_sources()
        self.compute_cell_contributions()

    def recompute_sources(self):
        """Recompute the what cells the point sources collide with.

        This function should be called if the mesh geometry has been modified.
        """

        # Determine what process owns a point and what cells it lies within
        mesh = self._function_space.mesh
        tol = float(1e2 * np.finfo(self._input_points.dtype).eps)
        if dolfinx.__version__ == "0.8.0":
            src_ranks, _, self._points, self._cells = (
                dolfinx.cpp.geometry.determine_point_ownership(
                    mesh._cpp_object, self._input_points, tol
                )
            )
            self._points = np.array(self._points).reshape(-1, 3)
        elif Version(dolfinx.__version__) >= Version("0.9.0.0"):
            collision_data = dolfinx.cpp.geometry.determine_point_ownership(
                mesh._cpp_object, self._input_points, tol
            )
            self._points = collision_data.dest_points
            self._cells = collision_data.dest_cells
            src_ranks = collision_data.src_owner
        else:
            raise NotImplementedError(f"Unsupported version of dolfinx: {dolfinx.__version__}")
        if -1 in src_ranks:
            raise ValueError("Point source is outside the mesh.")

    def compute_cell_contributions(self):
        """Compute the basis function values at the point sources."""
        mesh = self._function_space.mesh
        # Pull owning points back to reference cell
        mesh_nodes = mesh.geometry.x
        cmap = mesh.geometry.cmap

        ref_x = np.zeros((len(self._cells), mesh.topology.dim), dtype=mesh.geometry.x.dtype)
        for i, (point, cell) in enumerate(zip(self._points, self._cells)):
            geom_dofs = mesh.geometry.dofmap[cell]
            ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

        # Create expression evaluating a trial function (i.e. just the basis function)
        u = ufl.TestFunction(self._function_space)
        bs = self._function_space.dofmap.bs
        num_dofs = self._function_space.dofmap.dof_layout.num_dofs * bs
        if len(self._cells) > 0:
            # NOTE: Expression lives on only this communicator rank
            expr = dolfinx.fem.Expression(u, ref_x, comm=MPI.COMM_SELF)
            all_values = expr.eval(mesh, self._cells)
            # Diagonalize values (num_cells, num_points, num_dofs, bs) -> (num_cells, num_dofs)
            basis_values = np.empty((len(self._cells), num_dofs), dtype=all_values.dtype)
            for i in range(len(self._cells)):
                basis_values[i] = sum(
                    [
                        all_values[i, i * num_dofs * bs : (i + 1) * num_dofs * bs][
                            j * num_dofs : (j + 1) * num_dofs
                        ]
                        for j in range(bs)
                    ]
                )
        else:
            basis_values = np.zeros((0, num_dofs), dtype=dolfinx.default_scalar_type)
        self._basis_values = basis_values

    def apply_to_vector(
        self, b: dolfinx.fem.Function | dolfinx.la.Vector | PETSc.Vec, recompute: bool = False
    ):
        """Apply the point sources to a vector.

        Args:
            b: The vector to apply the point sources to.
            recompute: If the point sources should be recomputed before applying.
                Recomputation should only be done if the mesh geometry has been modified.

        Note:
            The user is responsible for forward scattering of the vector after
            applying the point sources.

        Note:
            If a PETSc vector is passed in, one has to call ``b.assemble()`` prior to solving the
            linear system (post scattering).
        """
        if recompute:
            self.recompute_sources()
            self.compute_cell_contributions()

        # Apply the point sources to the vector
        _dofs = self._function_space.dofmap.list[self._cells]
        unrolled_dofs = unroll_dofmap(_dofs, self._function_space.dofmap.bs)
        for dofs, values in zip(unrolled_dofs, self._basis_values, strict=True):
            if isinstance(b, dolfinx.fem.Function):
                b.x.array[dofs] += values * self._magnitude
            elif isinstance(b, dolfinx.la.Vector):
                b.array[dofs] += values * self._magnitude
            elif isinstance(b, PETSc.Vec):
                b.setValuesLocal(
                    dofs,
                    values * self._magnitude,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )
