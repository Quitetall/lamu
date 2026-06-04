"""widgetlib — a tiny demo package with a deliberate off-by-one bug.

The public surface is `window_sums`, used by downstream code to compute the
sum of each fixed-width window over a sequence. It currently drops the last
element of every window (classic off-by-one in the slice bound).
"""
from .core import window_sums

__all__ = ["window_sums"]
