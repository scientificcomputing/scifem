# Equivalence tests for the sorted-search helpers in script.py
# SPDX-License-Identifier: MIT

"""The dense reference implementations, kept side by side with the fast ones.

`find_position` and `compute_insert_position` used to build a dense
``len(data) x len(values)`` boolean matrix. They now use ``argsort`` + ``searchsorted``.
The originals are reproduced here verbatim as the specification: the tests assert the new
implementations agree with them on random input, so the rewrite cannot drift, and anyone
reading this file can see exactly what the fast versions are supposed to compute.

    python3 -m pytest test_helpers.py
"""

import numpy as np
import pytest

from script import (
    compute_insert_position,
    find_position,
    gather_ragged,
    unroll_insert_position,
)


# --------------------------------------------------------------------------- #
# the dense implementations these replaced
# --------------------------------------------------------------------------- #


def find_position_dense(data, values):
    if len(data) == 0:
        return np.zeros(0, dtype=np.int32)
    matches = values == data[:, None]
    if not np.all(np.any(matches, axis=1)):
        raise ValueError("find_position: data contains values not present in values")
    return matches.argmax(1)


def compute_insert_position_dense(data_owner, destination_ranks, out_size):
    process_pos_indicator = data_owner.reshape(-1, 1) == destination_ranks

    send_offsets = np.zeros(len(out_size) + 1, dtype=np.intc)
    send_offsets[1:] = np.cumsum(out_size)
    assert send_offsets[-1] == len(data_owner)

    proc_row, proc_col = np.nonzero(process_pos_indicator)
    cum_pos = np.cumsum(process_pos_indicator, axis=0)
    insert_position = cum_pos[proc_row, proc_col] - 1

    insert_position += send_offsets[proc_col]
    return insert_position


# --------------------------------------------------------------------------- #
# the worked examples from the docstrings
# --------------------------------------------------------------------------- #


def test_find_position_docstring_example():
    values = np.array([4, 5, 1, 3, 2], dtype=np.int32)
    data = np.array([1, 2, 3, 4, 5, 2, 1], dtype=np.int32)
    assert np.array_equal(find_position(data, values), [2, 4, 3, 0, 1, 4, 2])


def test_compute_insert_position_docstring_example():
    data_owner = np.array([0, 1, 1, 0, 2, 3], dtype=np.int32)
    destination_ranks = np.array([2, 0, 3, 1], dtype=np.int32)
    out_size = np.array([1, 2, 1, 2], dtype=np.int32)
    got = compute_insert_position(data_owner, destination_ranks, out_size)
    assert np.array_equal(got, [1, 4, 5, 2, 0, 3])


def test_unroll_insert_position_docstring_example():
    insert_position = np.array([1, 4, 5, 2, 0, 3], dtype=np.int32)
    expected = [3, 4, 5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 0, 1, 2, 9, 10, 11]
    assert np.array_equal(unroll_insert_position(insert_position, 3), expected)


def test_gather_ragged_docstring_example():
    offsets = np.array([0, 2, 2, 5], dtype=np.int64)
    positions, sizes = gather_ragged(offsets, np.array([2, 0], dtype=np.int64))
    assert np.array_equal(positions, [2, 3, 4, 0, 1])
    assert np.array_equal(sizes, [3, 2])


def test_gather_ragged_takes_an_empty_group():
    offsets = np.array([0, 2, 2, 5], dtype=np.int64)
    positions, sizes = gather_ragged(offsets, np.array([1, 1], dtype=np.int64))
    assert len(positions) == 0 and np.array_equal(sizes, [0, 0])


def test_gather_ragged_takes_an_empty_selection():
    offsets = np.array([0, 2, 2, 5], dtype=np.int64)
    positions, sizes = gather_ragged(offsets, np.zeros(0, dtype=np.int64))
    assert len(positions) == 0 and len(sizes) == 0


@pytest.mark.parametrize("seed", range(25))
def test_gather_ragged_matches_the_loop(seed):
    """The loop the vectorised form replaces, on ragged arrays with empty groups in them."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(0, 4, size=rng.integers(1, 12))
    offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    data = rng.integers(0, 100, size=int(offsets[-1]))
    selection = rng.integers(0, len(sizes), size=rng.integers(0, 15))

    expected = [v for i in selection for v in data[offsets[i] : offsets[i + 1]]]
    positions, got_sizes = gather_ragged(offsets, selection)
    assert np.array_equal(data[positions], expected)
    assert np.array_equal(got_sizes, sizes[selection])


# --------------------------------------------------------------------------- #
# equivalence with the dense implementations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(25))
def test_find_position_matches_dense(seed):
    rng = np.random.default_rng(seed)
    n_values = int(rng.integers(1, 40))
    # a permutation, so `values` has no repeats -- the usual case
    values = rng.permutation(n_values).astype(np.int32)
    data = rng.choice(values, size=int(rng.integers(0, 60))).astype(np.int32)
    assert np.array_equal(
        find_position(data, values), find_position_dense(data, values)
    )


@pytest.mark.parametrize("seed", range(25))
def test_find_position_matches_dense_with_repeats(seed):
    """Where `values` repeats, both must return the *first* occurrence."""
    rng = np.random.default_rng(1000 + seed)
    values = rng.integers(0, 6, size=int(rng.integers(2, 30))).astype(np.int32)
    data = rng.choice(values, size=int(rng.integers(1, 40))).astype(np.int32)
    assert np.array_equal(
        find_position(data, values), find_position_dense(data, values)
    )


def test_find_position_empty():
    values = np.array([3, 1, 2], dtype=np.int32)
    assert find_position(np.zeros(0, dtype=np.int32), values).size == 0


def test_find_position_rejects_missing_value():
    values = np.array([4, 5, 1], dtype=np.int32)
    data = np.array([4, 9], dtype=np.int32)
    with pytest.raises(ValueError, match="not present"):
        find_position(data, values)
    # and the value being larger than every entry is the same error, not an index error
    with pytest.raises(ValueError, match="not present"):
        find_position(np.array([99], dtype=np.int32), values)


@pytest.mark.parametrize("seed", range(40))
def test_compute_insert_position_matches_dense(seed):
    rng = np.random.default_rng(2000 + seed)
    n_ranks = int(rng.integers(1, 8))
    destination_ranks = rng.permutation(n_ranks).astype(np.int32)
    # a random multiset of destinations, then the counts they imply
    data_owner = rng.choice(destination_ranks, size=int(rng.integers(1, 80)))
    data_owner = data_owner.astype(np.int32)
    out_size = np.array(
        [np.count_nonzero(data_owner == r) for r in destination_ranks], dtype=np.int32
    )
    got = compute_insert_position(data_owner, destination_ranks, out_size)
    expected = compute_insert_position_dense(data_owner, destination_ranks, out_size)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("seed", range(15))
def test_compute_insert_position_is_a_permutation(seed):
    """The result has to be a permutation of range(n), or packing would drop entries."""
    rng = np.random.default_rng(3000 + seed)
    n_ranks = int(rng.integers(1, 6))
    destination_ranks = rng.permutation(n_ranks).astype(np.int32)
    data_owner = rng.choice(destination_ranks, size=int(rng.integers(1, 50)))
    data_owner = data_owner.astype(np.int32)
    out_size = np.array(
        [np.count_nonzero(data_owner == r) for r in destination_ranks], dtype=np.int32
    )
    got = compute_insert_position(data_owner, destination_ranks, out_size)
    assert np.array_equal(np.sort(got), np.arange(len(data_owner)))


def test_compute_insert_position_empty():
    got = compute_insert_position(
        np.zeros(0, dtype=np.int32),
        np.array([2, 0], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
    )
    assert got.size == 0


def test_compute_insert_position_groups_by_destination():
    """Items for the same destination land in one contiguous block, in input order."""
    data_owner = np.array([7, 3, 7, 3, 7], dtype=np.int32)
    destination_ranks = np.array([3, 7], dtype=np.int32)
    out_size = np.array([2, 3], dtype=np.int32)
    pos = compute_insert_position(data_owner, destination_ranks, out_size)
    packed = np.empty(5, dtype=np.int32)
    packed[pos] = np.arange(5, dtype=np.int32)
    # destination 3 first (its two items, in order), then destination 7 (its three)
    assert np.array_equal(packed, [1, 3, 0, 2, 4])
