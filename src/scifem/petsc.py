import dolfinx
import typing

__all__ = [
    "zero_petsc_vector",
    "ghost_update",
    "apply_lifting_and_set_bc",
]

try:
    from petsc4py import PETSc

    def zero_petsc_vector(b):
        """Zero a PETSc vector, including ghosts"""

        if b.getType() == PETSc.Vec.Type.NEST:
            for b_sub in b.getNestSubVecs():
                with b_sub.localForm() as b_local:
                    b_local.set(0.0)
        else:
            with b.localForm() as b_loc:
                b_loc.set(0)

    def ghost_update(x, insert_mode, scatter_mode):
        """Ghost update a vector"""
        if x.getType() == PETSc.Vec.Type.NEST:
            for x_sub in x.getNestSubVecs():
                x_sub.ghostUpdate(addv=insert_mode, mode=scatter_mode)
        else:
            x.ghostUpdate(addv=insert_mode, mode=scatter_mode)

except ImportError:

    def zero_petsc_vector(_b):
        """Zero a PETSc vector, including ghosts"""
        raise RuntimeError("petsc4py is not available. Cannot zero vector.")

    def ghost_update(_x, _insert_mode, _scatter_mode):
        """Ghost update a vector"""
        raise RuntimeError("petsc4py is not available. Cannot ghost update vector.")


if dolfinx.has_petsc4py:
    from petsc4py import PETSc
    import dolfinx.fem.petsc

    def apply_lifting_and_set_bc(
        b: PETSc.Vec,
        a: typing.Union[
            typing.Iterable[dolfinx.fem.Form], typing.Iterable[typing.Iterable[dolfinx.fem.Form]]
        ],
        bcs: typing.Union[
            typing.Iterable[dolfinx.fem.DirichletBC],
            typing.Iterable[typing.Iterable[dolfinx.fem.DirichletBC]],
        ],
        x: typing.Optional[PETSc.Vec] = None,
        alpha: float = 1.0,
    ):
        """Apply lifting to a vector and set boundary conditions.

        Convenience function to apply lifting and set boundary conditions for multiple matrix types.
        This modifies the vector b such that

        .. math::
            b_{free} = b - alpha a[i] (u_bc[i] - x[i])
            b_{bc} = u_bc[i]

        where :math:`b_{free}` is the free part of the vector, :math:`b_{bc}` is the part that has
        boundary conditions applied.

        Args:
            b: The vector to apply lifting to.
            a: Sequence fo forms to apply lifting from. If the system is blocked or nested,
                his is a nested list of forms.
            bcs: The boundary conditions to apply. If the form is blocked or nested, this is a list,
                while if it is a single form, this is a nested list.
            x: Vector to subtract from the boundary conditions. Usually used in a Newton iteration.
            alpha: The scaling factor for the boundary conditions.
        """
        if hasattr(dolfinx.fem.petsc, "create_vector_nest"):
            raise NotImplementedError(
                "This function is only implemented for later versions of DOLFINx."
            )

        try:
            bcs0 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(a, 0), bcs)
            bcs1 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(a, 1), bcs)
        except AssertionError:
            bcs0 = bcs
            bcs1 = bcs

        dolfinx.fem.petsc.apply_lifting(b, a, bcs=bcs1, x0=x, alpha=alpha)
        ghost_update(b, PETSc.InsertMode.ADD_VALUES, PETSc.ScatterMode.REVERSE)
        try:
            dolfinx.fem.petsc.set_bc(b, bcs0, x0=x, alpha=alpha)
        except AttributeError:
            for _bcs in bcs0:
                dolfinx.fem.petsc.set_bc(b, _bcs, x0=x, alpha=alpha)
        except TypeError:
            # Catch the case when bcs0 shouldn't be a nested list (e.g., for a single form)
            assert len(bcs0) == 1, "bcs0 should be a single DirichletBC or a list of DirichletBCs."
            dolfinx.fem.petsc.set_bc(b, bcs0[0], x0=x, alpha=alpha)

        ghost_update(b, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

else:

    def apply_lifting_and_set_bc(_b, _a, _bcs, _x, _alpha):  # type: ignore
        raise RuntimeError(
            "petsc4py is not available. Cannot apply lifting and set boundary conditions."
        )
