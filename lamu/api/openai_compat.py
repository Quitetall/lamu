"""OpenAI-compatible HTTP API layer.

Translates /v1/chat/completions → internal router → backend.
Always strips reasoning from `content` field; optionally returns in `reasoning_content`.
"""
from __future__ import annotations

import json
import time
import urllib.error
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from lamu.api.errors import backend_error_response, no_backend_response
from lamu.api.metrics import LamuMetrics
from lamu.core.errors import ReasoningOverflow
from lamu.core.health import HealthRegistry
from lamu.core.reasoning import get_extractor
from lamu.core.registry import load_registry
from lamu.core.router import Router
from lamu.core.scheduler import VramScheduler
from lamu.core.types import Capability, ModelEntry


# Errors expected from a backend over HTTP. Anything outside this set is a
# real bug and must propagate.
_BACKEND_HTTP_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    ConnectionError,
    TimeoutError,
    OSError,
    json.JSONDecodeError,
    KeyError,
    IndexError,
)

# Errors expected when probing a possibly-not-running service for auto-register.
_PROBE_EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    urllib.error.URLError,
    ConnectionError,
    TimeoutError,
    OSError,
    json.JSONDecodeError,
    KeyError,
    IndexError,
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[Message]
    max_tokens: int = 16384
    temperature: float = 0.7
    stream: bool = False
    top_k: Optional[int] = None
    top_p: Optional[float] = None


def create_app(
    scheduler: VramScheduler,
    registry: list[ModelEntry],
    health: Optional[HealthRegistry] = None,
) -> FastAPI:
    """Create the OpenAI-compatible FastAPI app."""

    app = FastAPI(title="LAMU", version="2.0")
    router = Router(scheduler, registry)
    entries_map: dict[str, ModelEntry] = {e.name: e for e in registry}
    health_reg = health or HealthRegistry()
    metrics = LamuMetrics()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "models_loaded": len(scheduler.loaded_models())}

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        # Refresh instantaneous gauges before serialising so the scrape
        # reflects the current scheduler + health snapshot.
        metrics.refresh(scheduler, health_reg)
        body, ctype = metrics.render()
        return Response(content=body, media_type=ctype)

    @app.get("/v1/models")
    async def list_models() -> dict:
        models_data = []
        for name, entry in entries_map.items():
            loaded = scheduler.is_loaded(name)
            models_data.append({
                "id": name,
                "object": "model",
                "owned_by": "local",
                "loaded": loaded,
                "params_b": entry.params_b,
                "vram_mb": entry.vram_mb,
                "capabilities": [c.value for c in entry.capabilities],
            })
        return {"data": models_data, "object": "list"}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: ChatRequest) -> JSONResponse | StreamingResponse:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        t_start = time.monotonic()

        # Route
        capabilities: Optional[list[Capability]] = None
        decision = router.route(
            model=req.model,
            capabilities=capabilities,
            health_map=health_reg.all() or None,
        )

        if not decision.model_name or not decision.loaded:
            metrics.requests_total.labels(
                model=req.model or "unknown", status="no_backend",
            ).inc()
            return no_backend_response(
                reason=f"No loaded model available: {decision.reason}",
            )

        loaded = scheduler.get_loaded(decision.model_name)
        if not loaded:
            return backend_error_response(
                reason="internal: model lost after routing",
                status_code=500,
            )

        scheduler.mark_used(decision.model_name)
        entry = entries_map.get(decision.model_name)
        extractor = get_extractor(entry.reasoning_marker if entry else None)

        # Build messages
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        # Forward to backend
        import urllib.request
        import urllib.error

        payload: dict = {
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": req.stream,
        }
        if req.top_k is not None:
            payload["top_k"] = req.top_k
        if req.top_p is not None:
            payload["top_p"] = req.top_p

        backend_url = f"http://localhost:{loaded.port}/v1/chat/completions"

        if req.stream:
            async def stream_gen() -> AsyncIterator[str]:
                try:
                    http_req = urllib.request.Request(
                        backend_url,
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(http_req, timeout=300) as resp:
                        in_reasoning = False
                        reasoning_done = False
                        open_tag = extractor.marker.open_tag if hasattr(extractor, '_marker') else "<think>"
                        close_tag = extractor.marker.close_tag if hasattr(extractor, '_marker') else "</think>"
                        pending = ""

                        for raw_line in resp:
                            line = raw_line.decode().strip()
                            if not line.startswith("data: "):
                                continue
                            chunk_str = line[6:]
                            if chunk_str == "[DONE]":
                                break

                            try:
                                delta = json.loads(chunk_str)["choices"][0]["delta"]
                                token = delta.get("content", "")
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

                            if not token:
                                continue

                            pending += token

                            # Strip reasoning in streaming mode
                            if not in_reasoning and not reasoning_done:
                                if open_tag in pending:
                                    in_reasoning = True
                                    pre = pending[:pending.index(open_tag)]
                                    pending = pending[pending.index(open_tag) + len(open_tag):]
                                    # Emit pre-content if any
                                    if pre.strip():
                                        chunk = _make_chunk(completion_id, created, decision.model_name, pre)
                                        yield f"data: {json.dumps(chunk)}\n\n"
                                elif len(pending) > len(open_tag) * 3:
                                    # No think block — emit as content
                                    reasoning_done = True
                                    chunk = _make_chunk(completion_id, created, decision.model_name, pending)
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    pending = ""

                            elif in_reasoning and not reasoning_done:
                                if close_tag in pending:
                                    reasoning_done = True
                                    in_reasoning = False
                                    pending = pending[pending.index(close_tag) + len(close_tag):]
                                    if pending.strip():
                                        chunk = _make_chunk(completion_id, created, decision.model_name, pending)
                                        yield f"data: {json.dumps(chunk)}\n\n"
                                    pending = ""
                                else:
                                    pending = ""  # discard reasoning tokens

                            elif reasoning_done:
                                chunk = _make_chunk(completion_id, created, decision.model_name, token)
                                yield f"data: {json.dumps(chunk)}\n\n"
                                pending = ""

                        # Flush
                        if pending.strip() and reasoning_done:
                            chunk = _make_chunk(completion_id, created, decision.model_name, pending)
                            yield f"data: {json.dumps(chunk)}\n\n"

                except _BACKEND_HTTP_ERRORS as e:
                    error_chunk = {"error": {"type": "backend_error", "message": str(e)}}
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                except ReasoningOverflow as e:
                    error_chunk = {
                        "error": {
                            "type": "reasoning_overflow",
                            "message": str(e),
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

                done_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": decision.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_gen(), media_type="text/event-stream")

        # Non-streaming
        try:
            http_req = urllib.request.Request(
                backend_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(http_req, timeout=300) as resp:
                data = json.loads(resp.read())

            msg = data["choices"][0]["message"]
            raw_content = msg.get("content") or ""
            reasoning_content = msg.get("reasoning_content", "")

            # Extract reasoning
            if reasoning_content:
                reasoning = reasoning_content
                content = raw_content
            else:
                reasoning, content = extractor.split(raw_content)

            # Build response (always strip reasoning from content field)
            usage = data.get("usage", {})
            timings = data.get("timings", {})

            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": decision.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": data["choices"][0].get("finish_reason", "stop"),
                }],
                "usage": usage,
            }

            # Qwen extension: reasoning_content field
            if reasoning:
                response["choices"][0]["message"]["reasoning_content"] = reasoning

            # Stats extension
            if timings:
                response["timings"] = timings

            # Metrics: success path
            metrics.requests_total.labels(
                model=decision.model_name, status="ok",
            ).inc()
            metrics.request_duration_seconds.labels(
                model=decision.model_name, phase="total",
            ).observe(time.monotonic() - t_start)
            metrics.tokens_generated_total.labels(
                model=decision.model_name, kind="content",
            ).inc(usage.get("completion_tokens", 0))
            if reasoning:
                metrics.tokens_generated_total.labels(
                    model=decision.model_name, kind="reasoning",
                ).inc(len(reasoning) // 4)  # rough char→token

            return JSONResponse(response)

        except urllib.error.URLError as e:
            # Record the failure against the model that was about to serve.
            health_reg.get_or_create(decision.model_name).record_error(e)
            metrics.requests_total.labels(
                model=decision.model_name, status="backend_error",
            ).inc()
            return backend_error_response(
                reason=f"Backend unreachable: {e}", status_code=502,
            )

    def _make_chunk(cid: str, created: int, model: str, content: str) -> dict:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }

    return app


def serve(port: int = 8020) -> None:
    """Start the OpenAI-compat server."""
    from lamu.core.scheduler import VramScheduler

    registry_path = Path.home() / "local-llm" / "config" / "models.yaml"
    entries = load_registry(registry_path)
    scheduler = VramScheduler()

    # Auto-register running models. Probe /v1/models to learn which model is
    # actually serving on each port, then match against the registry by
    # bidirectional substring (model id may be lowercased / suffixed).
    import json
    import urllib.request

    for probe_port in [8020, 8001]:
        try:
            req = urllib.request.Request(f"http://localhost:{probe_port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
            model_id = data["data"][0]["id"].lower()
        except _PROBE_EXPECTED_ERRORS:
            # Probe of a possibly-down backend — expected. Anything else is a
            # real bug and must propagate.
            continue

        for entry in entries:
            ename = entry.name.lower()
            if ename in model_id or model_id in ename:
                if not scheduler.is_loaded(entry.name):
                    scheduler.register_loaded(
                        entry, pid=0, port=probe_port, vram_actual_mb=entry.vram_mb
                    )
                break

    app = create_app(scheduler, entries)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
