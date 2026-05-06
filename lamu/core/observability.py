"""Single funnel for structured events. Stderr by default; optional file sink.

Every place in the codebase that needs to emit a structured event for
operators or downstream tooling must call `emit()` here. Direct
`print(json.dumps(...), file=sys.stderr)` calls bypass the file sink
and (later) the OTLP exporter — keep them out of new code.

Sinks resolved at call time (not at import) so tests + env-var swaps
work without re-importing:

  - LAMU_EVENT_LOG=/path/to/jsonl — append every event to a file
  - default — stderr only

Trace IDs are first-class. Pass `trace_id=` to thread an MCP request /
HTTP traceparent through every event spanned by that work.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional


def emit(event: str, *, trace_id: Optional[str] = None, **fields: object) -> None:
    """Emit a structured event.

    Stderr always gets a single JSON line. If LAMU_EVENT_LOG is set, the
    same line is appended to that file (best-effort — file errors are
    swallowed so the runtime never wedges on a bad sink).
    """
    payload: dict[str, object] = {"event": event}
    if trace_id is not None:
        payload["trace_id"] = trace_id
    payload.update(fields)
    line = json.dumps(payload, default=str, sort_keys=True)

    print(line, file=sys.stderr, flush=True)

    log_path = os.environ.get("LAMU_EVENT_LOG")
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            # File sink failures must never block an event — operators get
            # the stderr copy regardless.
            pass


def new_trace_id() -> str:
    """Generate a 16-hex-char trace id. Compatible with W3C TraceContext
    middle 16 chars; fine as a standalone identifier when no traceparent
    is in scope.
    """
    import uuid
    return uuid.uuid4().hex[:16]
