"""HIDDEN test for the offbyone_sum task.

The agent never sees this file. The harness copies it into the worktree only at
scoring time, runs it, and treats the exit code as ground truth — the agent
cannot fabricate a pass because we run this ourselves on the post-run worktree.

It pins the documented contract of `widgetlib.window_sums` with several
windows, so the buggy `i+size-1` slice fails and only the correct `i+size`
slice passes.
"""
import widgetlib


def test_basic_pairs():
    assert widgetlib.window_sums([1, 2, 3, 4], 2) == [3, 5, 7]


def test_window_of_three():
    assert widgetlib.window_sums([1, 2, 3, 4, 5], 3) == [6, 9, 12]


def test_full_window():
    # size == len(values): one window covering everything.
    assert widgetlib.window_sums([2, 4, 6], 3) == [12]


def test_size_one_is_identity():
    assert widgetlib.window_sums([5, 7, 9], 1) == [5, 7, 9]


def test_output_length_contract():
    vals = [10, 20, 30, 40, 50]
    out = widgetlib.window_sums(vals, 2)
    assert len(out) == len(vals) - 2 + 1
