# Create a point source for Poisson problem
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import TYPE_CHECKING

from mpi4py import MPI

if TYPE_CHECKING:
    from petsc4py import PETSc

try:
    from petsc4py import PETSc

    _HAS_PETSC = True
except ImportError:
    _HAS_PETSC = False

import dolfinx
import fenicsx_compat
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
        tol: np.floating | None = None,
    ) -> None:
        """Initialize a point source.

        Args:
            V: The function space the point source is defined in.
            points: The points where the point source is located.
                Input shape: ``(num_points, 3)``
            magnitude: The magnitudes of the point sources.
            tol: Tolerance for point location. If `None` 100 times machine epsilon is used.

        Note:
            Points should only be defined on one process. If they are sent in
            from multiple processes, multiple point sources will be created.

        Note:
            If the point source is outside the mesh, a ``ValueError`` will be raised.
        """
        self._function_space = V
        if points.ndim == 1:
            points = points.reshape(1, -3)
        elif points.ndim == 2 and points.shape[1] != 3:
            raise ValueError(f"Points should have shape (num_points, 3), got {points.shape}")

        self._input_points = points
        self._magnitude = magnitude
        # Initialize empty arrays
        self._points = np.empty((0, 3), dtype=points.dtype)
        self._cells = np.empty(0, dtype=np.int32)
        num_dofs = self._function_space.dofmap.dof_layout.num_dofs * self._function_space.dofmap.bs
        self._basis_values = np.empty(
            (0, num_dofs), dtype=self._function_space.mesh.geometry.x.dtype
        )
        self._tol = (
            float(1e2 * np.finfo(self._input_points.dtype).eps) if tol is None else float(tol)
        )
        self.recompute_sources()
        self.compute_cell_contributions()

    def recompute_sources(self, tol: float | None = None):
        """Recompute the what cells the point sources collide with.

        This function should be called if the mesh geometry has been modified.

        Args:
            tol: Tolerance for point location. If `None`, the tolerance from initialization is used.
        """

        # Determine what process owns a point and what cells it lies within
        mesh = self._function_space.mesh
        tol = self._tol if tol is None else tol
        # Points are float32 or float64; the constrained TypeVar cannot take the union
        collision_data = dolfinx.geometry.determine_point_ownership(  # type: ignore[type-var]
            mesh, self._input_points, padding=tol
        )
        self._points = collision_data.dest_points
        self._cells = collision_data.dest_cells
        src_ranks = collision_data.src_owner
        if -1 in src_ranks:
            raise ValueError("Point source is outside the mesh.")

    def compute_cell_contributions(self):
        """Compute the basis function values at the point sources."""
        mesh = self._function_space.mesh
        # Pull owning points back to reference cell
        mesh_nodes = mesh.geometry.x
        cmap = fenicsx_compat.cmap(mesh)

        ref_x = np.zeros((len(self._cells), mesh.topology.dim), dtype=mesh.geometry.x.dtype)
        for i, (point, cell) in enumerate(zip(self._points, self._cells)):
            geom_dofs = fenicsx_compat.dofmap(mesh)[cell]
            ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

        # Create expression evaluating a trial function (i.e. just the basis function)
        u = ufl.TestFunction(self._function_space)
        bs = self._function_space.dofmap.bs
        num_dofs = self._function_space.dofmap.dof_layout.num_dofs * bs
        if len(self._cells) > 0:
            # NOTE: Expression lives on only this communicator rank
            # Expression is evaluated for every point in every cell, which means that we
            # need to discard values that are not on the diagonal.
            expr = dolfinx.fem.Expression(u, ref_x, comm=MPI.COMM_SELF)
            all_values = expr.eval(mesh, self._cells)
            # Diagonalize values (num_cells, num_points, num_dofs, bs) -> (num_cells, num_dofs)
            basis_values = np.zeros((len(self._cells), num_dofs), dtype=dolfinx.default_scalar_type)
            if bs == 1:
                # Values have shape (num_cells, num_points, num_dofs)
                for i in range(len(self._cells)):
                    basis_values[i] = all_values[i, i, :]
            else:
                # Values have shape (num_cells, num_points, bs, num_dofs*bs)
                for i in range(len(self._cells)):
                    basis_values[i] = sum(all_values[i, i, j, :] for j in range(bs))
        else:
            basis_values = np.zeros((0, num_dofs), dtype=dolfinx.default_scalar_type)
        self._basis_values = basis_values

    def apply_to_vector(
        self, b: dolfinx.fem.Function | dolfinx.la.Vector | "PETSc.Vec", recompute: bool = False
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
            elif _HAS_PETSC and isinstance(b, PETSc.Vec):
                b.setValuesLocal(
                    dofs,
                    values * self._magnitude,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )
            else:
                raise TypeError(f"Unsupported vector type {type(b)}.")
