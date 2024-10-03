from __future__ import annotations

from mpi4py import MPI
import ufl
import numpy as np
import dolfinx


def assemble_scalar(J: ufl.form.Form | dolfinx.fem.Form) -> np.floating | np.complexfloating:
    """Assemble a scalar form and gather result across processes

    Args:
        form: The form to assemble.

    Returns:
        The accumulated value of the assembled form.
    """
    compiled_form = dolfinx.fem.form(J)
    if (rank := compiled_form.rank) != 0:
        raise ValueError(f"Form must be a scalar form, got for of arity {rank}")
    local_result = dolfinx.fem.assemble_scalar(compiled_form)
    return compiled_form.mesh.comm.allreduce(local_result, op=MPI.SUM)
