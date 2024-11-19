from __future__ import annotations

from typing import Callable
import logging
from packaging.version import parse as _v

import numpy as np
from petsc4py import PETSc
import ufl
import dolfinx


__all__ = ["NewtonSolver", "BlockedNewtonSolver"]

logger = logging.getLogger(__name__)

# assemble_vector_block(scale=...) is renamed assemble_vector_block(alpha=...)
# in 0.9
_alpha_kw: str = "alpha"
if _v(dolfinx.__version__) < _v("0.9"):
    _alpha_kw = "scale"


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
        self.b = dolfinx.fem.petsc.create_vector_block(self._F)
        self.A = dolfinx.fem.petsc.create_matrix_block(self._J)
        self.dx = dolfinx.fem.petsc.create_vector_block(self._F)
        self.x = dolfinx.fem.petsc.create_vector_block(self._F)

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
            constants_L = [
                form and dolfinx.cpp.fem.pack_constants(form._cpp_object) for form in self._F
            ]
            coeffs_L = [dolfinx.cpp.fem.pack_coefficients(form._cpp_object) for form in self._F]
            constants_a = [
                [
                    dolfinx.cpp.fem.pack_constants(form._cpp_object)
                    if form is not None
                    else np.array([], dtype=PETSc.ScalarType)
                    for form in forms
                ]
                for forms in self._J
            ]
            coeffs_a = [
                [
                    {} if form is None else dolfinx.cpp.fem.pack_coefficients(form._cpp_object)
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
            self.b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(
                self.A, self._J, bcs=self.bcs, constants=constants_a, coeffs=coeffs_a
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
        return self

    def __del__(self):
        """Clean up the solver by destroying PETSc objects."""
        self.A.destroy()
        self.b.destroy()
        self.dx.destroy()
        self._solver.destroy()
        self.x.destroy()


class BlockedNewtonSolver(dolfinx.cpp.nls.petsc.NewtonSolver):
    def __init__(
        self,
        F: list[ufl.form.Form],
        u: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC] = [],
        J: list[list[ufl.form.Form]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        petsc_options: dict | None = None,
        entity_maps: dict | None = None,
    ):
        """Initialize solver for solving a non-linear problem using Newton's method.
        Args:
            F: List of PDE residuals [F_0(u, v_0), F_1(u, v_1), ...]
            u: List of unknown functions u=[u_0, u_1, ...]
            bcs: List of Dirichlet boundary conditions
            J: UFL representation of the Jacobian (Optional)
                Note:
                    If not provided, the Jacobian will be computed using the
                    assumption that the test functions come from a ``ufl.MixedFunctionSpace``
            form_compiler_options: Options used in FFCx
                compilation of this form. Run ``ffcx --help`` at the
                command line to see all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See ``python/dolfinx/jit.py``
                for all available options. Takes priority over all
                other option values.
            petsc_options:
                Options passed to the PETSc Krylov solver.
            entity_maps: Maps used to map entities between different meshes.
                Only needed if the forms have not been compiled a priori,
                and has coefficients, test, or trial functions that are defined on different meshes.
        """
        # Initialize base class
        super().__init__(u[0].function_space.mesh.comm)

        # Set PETSc options for Krylov solver
        prefix = self.krylov_solver.getOptionsPrefix()
        if prefix is None:
            prefix = ""
        if petsc_options is not None:
            # Set PETSc options
            opts = PETSc.Options()
            opts.prefixPush(prefix)
            for k, v in petsc_options.items():
                opts[k] = v
            opts.prefixPop()
            self.krylov_solver.setFromOptions()
        self._F = dolfinx.fem.form(
            F,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        # Create the Jacobian matrix, dF/du
        if J is None:
            if _v(dolfinx.__version__) < _v("0.9"):
                raise RuntimeError(
                    "Automatic computation of Jacobian for blocked problem is only"
                    + "supported in DOLFINx 0.9 and later"
                )
            du = ufl.TrialFunctions(ufl.MixedFunctionSpace(*[ui.function_space for ui in u]))
            J = ufl.extract_blocks(sum(ufl.derivative(sum(F), u[i], du[i]) for i in range(len(u))))
        self._a = dolfinx.fem.form(
            J,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )

        self._bcs = bcs
        self._u = u
        self._pre_solve_callback: Callable[["BlockedNewtonSolver"], None] | None = None
        self._post_solve_callback: Callable[["BlockedNewtonSolver"], None] | None = None

        # Create structures for holding arrays and matrix
        self._b = dolfinx.fem.petsc.create_vector_block(self._F)
        self._J = dolfinx.fem.petsc.create_matrix_block(self._a)
        self._dx = dolfinx.fem.petsc.create_vector_block(self._F)
        self._x = dolfinx.fem.petsc.create_vector_block(self._F)
        self._J.setOptionsPrefix(prefix)
        self._J.setFromOptions()

        self.setJ(self._assemble_jacobian, self._J)
        self.setF(self._assemble_residual, self._b)
        self.set_form(self._pre_newton_iteration)
        self.set_update(self._update_function)

    def set_pre_solve_callback(self, callback: Callable[["BlockedNewtonSolver"], None]):
        """Set a callback function that is called before each Newton iteration."""
        self._pre_solve_callback = callback

    def set_post_solve_callback(self, callback: Callable[["BlockedNewtonSolver"], None]):
        """Set a callback function that is called after each Newton iteration."""
        self._post_solve_callback = callback

    @property
    def L(self) -> list[dolfinx.fem.Form]:
        """Compiled linear form (the residual form)"""
        return self._F

    @property
    def a(self) -> list[list[dolfinx.fem.Form]]:
        """Compiled bilinear form (the Jacobian form)"""
        return self._a

    @property
    def u(self):
        return self._u

    def __del__(self):
        self._J.destroy()
        self._b.destroy()
        self._dx.destroy()
        self._x.destroy()

    def _pre_newton_iteration(self, x: PETSc.Vec) -> None:
        """Function called before the residual or Jacobian is
        computed.
        Args:
           x: The vector containing the latest solution
        """
        if self._pre_solve_callback is not None:
            self._pre_solve_callback(self)
        # Scatter previous solution `u=[u_0, ..., u_N]` to `x`; the
        # blocked version used for lifting
        dolfinx.cpp.la.petsc.scatter_local_vectors(
            x,
            [ui.x.petsc_vec.array_r for ui in self._u],
            [
                (
                    ui.function_space.dofmap.index_map,
                    ui.function_space.dofmap.index_map_bs,
                )
                for ui in self._u
            ],
        )
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def _assemble_residual(self, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.
        Args:
            x: The vector containing the latest solution
            b: Vector to assemble the residual into
        """
        # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector_block(
            b,
            self._F,
            self._a,
            bcs=self._bcs,
            x0=x,
            # dolfinx 0.8 compatibility
            # this is called 'scale' in 0.8, 'alpha' in 0.9
            **{_alpha_kw: -1.0},
        )
        b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

    def _assemble_jacobian(self, x: PETSc.Vec, A: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.
        Args:
            x: The vector containing the latest solution
        """
        # Assemble Jacobian
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(A, self._a, bcs=self._bcs)
        A.assemble()

    def _update_function(self, solver, dx: PETSc.Vec, x: PETSc.Vec):
        if self._post_solve_callback is not None:
            self._post_solve_callback(self)
        # Update solution
        offset_start = 0
        for ui in self._u:
            Vi = ui.function_space
            num_sub_dofs = Vi.dofmap.index_map.size_local * Vi.dofmap.index_map_bs
            ui.x.petsc_vec.array_w[:num_sub_dofs] -= (
                self.relaxation_parameter * dx.array_r[offset_start : offset_start + num_sub_dofs]
            )
            ui.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset_start += num_sub_dofs

    def solve(self):
        """Solve non-linear problem into function. Returns the number
        of iterations and if the solver converged."""
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        n, converged = super().solve(self._x)
        self._x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return n, converged
