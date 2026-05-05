"""HTTP error helpers for the OpenAI-compat layer.

Every routing/backend failure flows through `no_backend_response` so clients
get a structured 503 with a Retry-After header — never a hang and never a
generic 500.
"""
from __future__ import annotations

from fastapi.responses import JSONResponse


def no_backend_response(reason: str, retry_after: int = 30) -> JSONResponse:
    """Build a 503 the client can act on.

    Args:
        reason: human-readable explanation (e.g. "no_healthy_backend").
        retry_after: seconds the client should wait before retrying.

    Returns:
        JSONResponse with status_code=503 and the Retry-After header set.
    """
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": str(retry_after)},
        content={
            "error": {
                "type": "backend_unavailable",
                "message": reason,
                "retry_after_s": retry_after,
            }
        },
    )


def backend_error_response(
    reason: str, status_code: int = 502,
) -> JSONResponse:
    """Backend reachable but request failed — 502 Bad Gateway by default."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": "backend_error",
                "message": reason,
            }
        },
    )
