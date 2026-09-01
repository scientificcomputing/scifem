from __future__ import annotations

import sys
from typing import Callable
import logging

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


from packaging.version import parse as _v

import numpy as np
import ufl
import dolfinx

from . import compat

__all__ = ["NewtonSolver"]

if dolfinx.has_petsc4py and dolfinx.has_petsc:
    logger = logging.getLogger(__name__)
    from petsc4py import PETSc
    import dolfinx.fem.petsc

    # assemble_vector_block(scale=...) is renamed assemble_vector_block(alpha=...)
    # in 0.9
    _alpha_kw: str = "alpha"
    if _v(dolfinx.__version__) < _v("0.9"):
        _alpha_kw = "scale"

    @deprecated("NewtonSolver is deprecated in favor of `dolfinx.fem.petsc.NonlinearProblem`.")
    class NewtonSolver:
        max_iterations: int
        _bcs: list[dolfinx.fem.DirichletBC]
        A: PETSc.Mat
        b: PETSc.Vec
        _J: dolfinx.fem.Form | ufl.form.Form
        _F: dolfinx.fem.Form | ufl.form.Form
        _dx: PETSc.Vec
        _error_on_convergence: bool
        _pre_solve_callback: Callable[["NewtonSolver"], None] | None
        _post_solve_callback: Callable[["NewtonSolver"], None] | None

        def __init__(
            self,
            F: list[dolfinx.fem.Form],
            J: list[list[dolfinx.fem.Form]],
            w: list[dolfinx.fem.Function],
            bcs: list[dolfinx.fem.DirichletBC] | None = None,
            max_iterations: int = 5,
            petsc_options: dict[str, str | float | int | None] | None = None,
            error_on_nonconvergence: bool = True,
        ):
            """
            Create a Newton solver for a block nonlinear problem ``F(u) = 0``.
            Solved as ``J(u) du = -F(u)``, where ``J`` is the Jacobian of ``F``.

            Args:
                F: List of forms defining the residual
                J: List of lists of forms defining the Jacobian
                w: List of functions representing the solution
                bcs: List of Dirichlet boundary conditions
                max_iterations: Maximum number of iterations in Newton solver
                petsc_options: PETSc options for Krylov subspace solver.
                error_on_nonconvergence: Raise an error if the linear solver
                    does not converge or Newton solver doesn't converge
            """
            # Compile forms if not compiled. Will throw error it requires entity maps
            self._F = dolfinx.fem.form(F)
            self._J = dolfinx.fem.form(J)

            # Store solution and accessible/modifiable properties
            self._pre_solve_callback = None
            self._post_solve_callback = None
            self.max_iterations = max_iterations
            self._error_on_convergence = error_on_nonconvergence
            self.bcs = [] if bcs is None else bcs
            self.w = w

            # Create PETSc objects for block assembly
            try:
                self.b = dolfinx.fem.petsc.create_vector_block(self._F)
                self.A = dolfinx.fem.petsc.create_matrix_block(self._J)
                self.dx = dolfinx.fem.petsc.create_vector_block(self._F)
                self.x = dolfinx.fem.petsc.create_vector_block(self._F)
            except AttributeError:
                self.b = dolfinx.fem.petsc.create_vector(
                    dolfinx.fem.extract_function_spaces(self._F), kind="mpi"
                )
                self.A = dolfinx.fem.petsc.create_matrix(self._J, kind="mpi")
                self.dx = dolfinx.fem.petsc.create_vector(
                    dolfinx.fem.extract_function_spaces(self._F), kind="mpi"
                )
                self.x = dolfinx.fem.petsc.create_vector(
                    dolfinx.fem.extract_function_spaces(self._F), kind="mpi"
                )

                self._maps = [compat.form_map(form) for form in self._F]

            # Set PETSc options
            opts = PETSc.Options()
            if petsc_options is not None:
                for k, v in petsc_options.items():
                    opts[k] = v

            # Define KSP solver
            self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
            self._solver.setOperators(self.A)
            self._solver.setFromOptions()

            # Set matrix and vector PETSc options
            self.A.setFromOptions()
            self.b.setFromOptions()

        def set_pre_solve_callback(self, callback: Callable[["NewtonSolver"], None]):
            """Set a callback function that is called before each Newton iteration."""
            self._pre_solve_callback = callback

        def set_post_solve_callback(self, callback: Callable[["NewtonSolver"], None]):
            """Set a callback function that is called after each Newton iteration."""
            self._post_solve_callback = callback

        def solve(self, atol=1e-6, rtol=1e-8, beta=1.0) -> int:
            """Solve the nonlinear problem using Newton's method.

            Args:
                atol: Absolute tolerance for the update.
                rtol: Relative tolerance for the update.
                beta: Damping parameter for the update.

            Returns:
                The number of Newton iterations used to converge.

            Note:
                The tolerance is on the 0-norm of the update.
            """
            i = 1

            while i <= self.max_iterations:
                if self._pre_solve_callback is not None:
                    self._pre_solve_callback(self)

                # Pack constants and coefficients
                try:
                    constants_L = dolfinx.fem.pack_constants(self._F)
                    coeffs_L = dolfinx.fem.pack_coefficients(self._F)
                    constants_a = [
                        dolfinx.fem.pack_constants(form)
                        if form is not None
                        else np.array([], dtype=self.x.array.dtype)
                        for form in self._J
                    ]
                    coeffs_a = [
                        {} if form is None else dolfinx.fem.pack_coefficients(form)
                        for form in self._J
                    ]
                except AttributeError:
                    # NOTE: DOLFINx 0.9 compatibility
                    # Pack constants and coefficients
                    constants_L = [
                        form and dolfinx.cpp.fem.pack_constants(form._cpp_object)
                        for form in self._F
                    ]
                    coeffs_L = [
                        dolfinx.cpp.fem.pack_coefficients(form._cpp_object) for form in self._F
                    ]
                    constants_a = [
                        [
                            dolfinx.cpp.fem.pack_constants(form._cpp_object)
                            if form is not None
                            else np.array([], dtype=self.x.array.dtype)
                            for form in forms
                        ]
                        for forms in self._J
                    ]
                    coeffs_a = [
                        [
                            {}
                            if form is None
                            else dolfinx.cpp.fem.pack_coefficients(form._cpp_object)
                            for form in forms
                        ]
                        for forms in self._J
                    ]
                # Scatter previous solution `w` to `self.x`, the blocked version used for lifting
                dolfinx.cpp.la.petsc.scatter_local_vectors(
                    self.x,
                    [si.x.petsc_vec.array_r for si in self.w],
                    [
                        (
                            si.function_space.dofmap.index_map,
                            si.function_space.dofmap.index_map_bs,
                        )
                        for si in self.w
                    ],
                )
                self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

                # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
                with self.b.localForm() as b_local:
                    b_local.set(0.0)
                try:
                    dolfinx.fem.petsc.assemble_vector_block(
                        self.b,
                        self._F,
                        self._J,
                        bcs=self.bcs,
                        x0=self.x,
                        coeffs_a=coeffs_a,
                        constants_a=constants_a,
                        coeffs_L=coeffs_L,
                        constants_L=constants_L,
                        # dolfinx 0.8 compatibility
                        # this is called 'scale' in 0.8, 'alpha' in 0.9
                        **{_alpha_kw: -1.0},
                    )
                except AttributeError:
                    dolfinx.fem.petsc.assemble_vector(
                        self.b, self._F, coeffs=coeffs_L, constants=constants_L
                    )
                    bcs1 = dolfinx.fem.bcs_by_block(
                        dolfinx.fem.extract_function_spaces(self._J, 1), self.bcs
                    )
                    dolfinx.fem.petsc.apply_lifting(
                        self.b,
                        self._J,
                        bcs=bcs1,
                        x0=self.x,
                        coeffs=coeffs_a,
                        constants=constants_a,
                        alpha=-1.0,
                    )
                    self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    bcs0 = dolfinx.fem.bcs_by_block(
                        dolfinx.fem.extract_function_spaces(self._F), self.bcs
                    )
                    dolfinx.fem.petsc.set_bc(self.b, bcs0, x0=self.x, alpha=-1.0)
                    self.b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

                # Assemble Jacobian
                self.A.zeroEntries()
                try:
                    dolfinx.fem.petsc.assemble_matrix_block(
                        self.A, self._J, bcs=self.bcs, constants=constants_a, coeffs=coeffs_a
                    )
                except AttributeError:
                    dolfinx.fem.petsc.assemble_matrix(
                        self.A, self._J, self.bcs, coeffs=coeffs_a, constants=constants_a
                    )
                self.A.assemble()

                self._solver.solve(self.b, self.dx)
                if self._error_on_convergence:
                    if (status := self._solver.getConvergedReason()) <= 0:
                        raise RuntimeError(f"Linear solver did not converge, got reason: {status}")

                # Update solution
                offset_start = 0
                for s in self.w:
                    num_sub_dofs = (
                        s.function_space.dofmap.index_map.size_local
                        * s.function_space.dofmap.index_map_bs
                    )
                    s.x.petsc_vec.array_w[:num_sub_dofs] -= (
                        beta * self.dx.array_r[offset_start : offset_start + num_sub_dofs]
                    )
                    s.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                    )
                    offset_start += num_sub_dofs

                if self._post_solve_callback is not None:
                    self._post_solve_callback(self)

                # Compute norm of update
                residual = self.dx.norm(PETSc.NormType.NORM_2)
                if i == 1:
                    self.residual_0 = residual
                relative_residual = residual / max(self.residual_0, atol)

                logger.info(
                    f"Newton iteration {i}"
                    f": r (abs) = {residual} (tol={atol}), "
                    f"r (rel) = {relative_residual} (tol={rtol})"
                )
                if relative_residual < rtol or residual < atol:
                    return i
                i += 1

            if self._error_on_convergence:
                raise RuntimeError("Newton solver did not converge")
            else:
                return self.max_iterations

        @property
        def F(self):
            """The list of residuals where each entry is a ``dolfinx.fem.Form``."""
            return self._F

        @property
        def J(self):
            """
            The Jacobian blocks represented as lists of lists where each entry
            is a ``dolfinx.fem.Form``.
            """
            return self._J

        def __del__(self):
            """Clean up the solver by destroying PETSc objects."""
            self.A.destroy()
            self.b.destroy()
            self.dx.destroy()
            self._solver.destroy()
            self.x.destroy()


else:

    class NewtonSolver:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "NewtonSolver is not available in this version of DOLFINx. "
                "Please install a version of DOLFINx with PETSc4py support."
            )
