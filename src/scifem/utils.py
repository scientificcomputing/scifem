from __future__ import annotations

import numpy.typing as npt
import numpy as np


def unroll_dofmap(dofs: npt.NDArray[np.int32], bs: int) -> npt.NDArray[np.int32]:
    """
    Given a two-dimensional dofmap of size `(num_cells, num_dofs_per_cell)`
    Expand the dofmap by its block size such that the resulting array
    is of size `(num_cells, bs*num_dofs_per_cell)`
    """
    num_cells, num_dofs_per_cell = dofs.shape
    unrolled_dofmap = np.repeat(dofs, bs).reshape(num_cells, num_dofs_per_cell * bs) * bs
    unrolled_dofmap += np.tile(np.arange(bs), num_dofs_per_cell)
    return unrolled_dofmap


def group_by_key(keys: npt.NDArray[np.integer]) -> list[tuple[int, npt.NDArray[np.intp]]]:
    """The distinct values of ``keys`` and, for each, the indices where it occurs.

    Args:
        keys: One integer key per entry.

    Returns:
        ``(key, indices)`` pairs, in increasing key order.
    """
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    groups = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])
    return list(zip(unique.tolist(), groups))
