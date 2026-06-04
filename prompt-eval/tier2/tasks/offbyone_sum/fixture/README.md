# widgetlib

A tiny utility package. The one public function is `widgetlib.window_sums`,
which returns the sum of each contiguous fixed-width window over a sequence.

```python
from widgetlib import window_sums
window_sums([1, 2, 3, 4], 2)  # -> [3, 5, 7]
```

Downstream callers depend on the documented contract above.
