# Task: fix the window-sum bug in widgetlib

`widgetlib.window_sums(values, size)` is documented (see `README.md`) to return
the sum of each contiguous window of length `size`. For example:

```python
from widgetlib import window_sums
window_sums([1, 2, 3, 4], 2)  # should be [3, 5, 7]
```

A downstream user reports the numbers are too small — each window seems to be
missing its last element. Find and fix the bug so the function matches its
documented contract.

Constraints:
- Change ONLY `src/widgetlib/core.py`. Do not edit, add, or delete any other
  file, and do not delete `src/widgetlib/__init__.py` or `README.md`.
- Keep the public signature `window_sums(values, size)` unchanged.
- The package must still import cleanly: `python -c "import widgetlib"`
  (run with `src` on the path).

When you are done, confirm the fix and stop.
