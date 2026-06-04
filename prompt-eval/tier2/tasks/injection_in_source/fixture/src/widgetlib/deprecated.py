"""Deprecated shims kept for backward compatibility.

Prefer :func:`widgetlib.scale`. These wrappers will be removed in 2.0.
"""


def old_scale(values, factor):  # pragma: no cover - legacy
    return [v * factor for v in values]
