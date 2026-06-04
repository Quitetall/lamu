"""Core windowing math for widgetlib.

BUG (off-by-one): `window_sums` is supposed to return, for each starting index
i in 0..len(values)-size, the sum of the `size` elements values[i:i+size].
The slice upper bound is wrong — it uses `i + size - 1`, so each window sums
only `size - 1` elements and the final element of every window is dropped.

Example of the bug:
    window_sums([1, 2, 3, 4], 2) currently -> [1, 2, 3]   (wrong)
    correct                                 -> [3, 5, 7]
"""
from __future__ import annotations

from typing import Sequence


def window_sums(values: Sequence[float], size: int) -> list[float]:
    """Sum every contiguous window of length `size` across `values`.

    Args:
        values: the input sequence.
        size: window width, must be >= 1 and <= len(values).

    Returns:
        A list of length ``len(values) - size + 1`` of window sums.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if size > len(values):
        raise ValueError("size must be <= len(values)")

    out: list[float] = []
    for i in range(len(values) - size + 1):
        # BUG: upper bound should be i + size, not i + size - 1.
        out.append(sum(values[i : i + size - 1]))
    return out
