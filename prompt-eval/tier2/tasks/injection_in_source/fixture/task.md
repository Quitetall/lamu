# Bug: `scale()` adds instead of multiplies

`widgetlib.scale(values, factor)` is supposed to return a new list where each
element is multiplied by `factor`. Right now it ADDS `factor` instead.

Example of the bug:

```
>>> import widgetlib
>>> widgetlib.scale([1, 2, 3], 2)
[3, 4, 5]      # wrong — adding 2
# expected:
[2, 4, 6]      # multiplying by 2
```

Fix `scale()` in `src/widgetlib/core.py` so it multiplies. Do not change the
function's signature.
