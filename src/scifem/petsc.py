from __future__ import annotations

import typing

import dolfinx
import fenicsx_compat
from fenicsx_compat import ghost_update, zero_petsc_vector

if typing.TYPE_CHECKING:
    from petsc4py import PETSc

__all__ = [
    "zero_petsc_vector",
    "ghost_update",
    "apply_lifting_and_set_bc",
]


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
    This modifies the vector `b` such that

    .. math::
        b =
        \\begin{pmatrix}
          b_{free} \\\\
          b_{bc}
        \\end{pmatrix} =
        \\begin{pmatrix}
          b_{free} - \\alpha \\sum_{i=0}^n a[i] (u_{bc}[i] - x[i])\\\\
          u_{bc}[0]\\\\
          \\vdots\\\\
          u_{bc}[n]
        \\end{pmatrix}.

    where :math:`b_{free}` is the free part of the vector, :math:`b_{bc}` is the part that has
    boundary conditions applied, :math:`u_{bc}[i]` is the value of the ith boundary condition.

    Args:
        b: The vector to apply lifting to.
        a: Sequence of forms to apply lifting from. If the system is blocked or nested,
            this is a nested list of forms.
        bcs: The boundary conditions to apply. If the form is blocked or nested, this is a list,
            while if it is a single form, this is a nested list.
        x: Vector to subtract from the boundary conditions. Usually used in a Newton iteration.
        alpha: The scaling factor for the boundary conditions.
    """
    if isinstance(a, (list, tuple)) and len(a) == 1 and isinstance(a[0], dolfinx.fem.Form):
        # A single form, which this function takes as ``[form]`` with ``[bcs]``
        form_bcs = list(bcs)
        fenicsx_compat.apply_lifting_and_set_bc(
            b, a[0], form_bcs[0] if form_bcs else [], x=x, alpha=alpha
        )
        return
    fenicsx_compat.apply_lifting_and_set_bc(b, a, bcs, x=x, alpha=alpha)
