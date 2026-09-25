"""Layer for small backward compatibility wrappers for UFL"""

import ufl

__all__ = ["apply_pullback_inverse"]


def _inverse_identity(expr: ufl.core.expr.Expr, domain: ufl.Mesh) -> ufl.core.expr.Expr:
    return expr


def _inverse_contravariant_piola(expr: ufl.core.expr.Expr, domain: ufl.Mesh) -> ufl.core.expr.Expr:
    """``u = J u_ref / det J`` inverted: ``u_ref = det J K u``."""
    return ufl.JacobianDeterminant(domain) * ufl.dot(ufl.JacobianInverse(domain), expr)


def _inverse_covariant_piola(expr: ufl.core.expr.Expr, domain: ufl.Mesh) -> ufl.core.expr.Expr:
    """``u = K^T u_ref`` inverted: ``u_ref = J^T u``."""
    return ufl.dot(ufl.transpose(ufl.Jacobian(domain)), expr)


_INVERSE_PULLBACKS = {
    "IdentityPullback": _inverse_identity,
    "ContravariantPiola": _inverse_contravariant_piola,
    "CovariantPiola": _inverse_covariant_piola,
}


def apply_pullback_inverse(
    pullback: ufl.pullback.AbstractPullback, expr: ufl.core.expr.Expr, domain: ufl.Mesh
) -> ufl.core.expr.Expr:
    """Map ``expr`` from the physical cell to the reference cell, across UFL versions.

    Uses ``pullback.apply_inverse``, added in https://github.com/FEniCS/ufl/pull/511, when the
    installed UFL has it, and otherwise a closed form for the identity, contravariant Piola and
    covariant Piola pullbacks.

    Args:
        pullback: The element's pullback.
        expr: A physical-cell expression.
        domain: The domain whose Jacobian relates the two cells.

    Returns:
        ``expr`` pulled back to the reference cell.

    Raises:
        NotImplementedError: If this UFL has no ``apply_inverse`` and there is no closed form
            for ``pullback`` here.
    """
    if hasattr(pullback, "apply_inverse"):
        return pullback.apply_inverse(expr, domain)
    name = type(pullback).__name__
    try:
        return _INVERSE_PULLBACKS[name](expr, domain)
    except KeyError:
        raise NotImplementedError(
            f"This UFL has no {name}.apply_inverse (added in FEniCS/ufl#511), and there is no "
            "closed form for it in scifem. Upgrade UFL."
        ) from None
