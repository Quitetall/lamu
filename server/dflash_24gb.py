"""
DFlash server for 24 GB GPUs (RTX 4090) with VRAM park/unpark dance.

The stock server.py loads target+draft eagerly (~20 GB) leaving no headroom
for the rollback cache. This wrapper starts the daemon with a small max-ctx,
parks/unparks around each request to fit within 24 GB.

    python server/dflash_24gb.py --port 8000

Protocol:
  1. Daemon boots with --max-ctx auto-fitted to prompt+gen+pad
  2. Each request: write prompt.bin + generate cmd → read streamed tokens
  3. Budget kept low (6-10) to minimize rollback cache (~0.5-1 GB vs 5 GB at 22)

OpenAI-compatible: /v1/chat/completions, /v1/models, /health
"""
import argparse
import asyncio
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DFLASH_DIR = ROOT / "lucebox-hub" / "dflash"

app = FastAPI()
MODEL_NAME = "dflash/qwen3.6-27b"

# Global state
daemon_proc = None
daemon_lock = None
r_pipe = None
tokenizer = None
stop_ids = set()


class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: list[dict]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


def get_tokenizer():
    global tokenizer, stop_ids
    if tokenizer is None:
        from transformers import AutoTokenizer
        tok_repo = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.6-27B")
        tokenizer = AutoTokenizer.from_pretrained(tok_repo, trust_remote_code=True)
        # Stop tokens
        for s in ["<|im_end|>", "<|endoftext|>"]:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_ids.add(ids[0])
    return tokenizer


def tokenize_to_file(text: str, path: str) -> list[int]:
    tok = get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return ids


def apply_chat_template(messages: list[dict]) -> str:
    tok = get_tokenizer()
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def start_daemon(target: str, draft: str, bin_path: str, budget: int, max_ctx: int):
    """Start dflash daemon with VRAM-safe settings for 24 GB."""
    global daemon_proc, daemon_lock, r_pipe

    daemon_lock = asyncio.Lock()
    r, w = os.pipe()

    # Resolve draft safetensors
    draft_p = Path(draft)
    if draft_p.is_dir():
        for st in draft_p.rglob("model.safetensors"):
            draft = str(st)
            break

    bin_abs = str(Path(bin_path).resolve())
    env = {
        **os.environ,
        # Q4_0 KV cache to save VRAM
        "DFLASH27B_KV_K": "q4_0",
        "DFLASH27B_KV_V": "q4_0",
        # No FA window (full attention on short ctx)
        "DFLASH27B_FA_WINDOW": "0",
        # Unified memory fallback
        "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
    }

    cmd = [
        bin_abs, target, draft, "--daemon",
        "--fast-rollback", "--ddtree",
        f"--ddtree-budget={budget}",
        f"--max-ctx={max_ctx}",
        f"--stream-fd={w}",
    ]

    print(f"[daemon] starting: budget={budget} max_ctx={max_ctx}", flush=True)
    print(f"[daemon] cmd: {' '.join(cmd[:6])}...", flush=True)

    daemon_proc = subprocess.Popen(
        cmd, pass_fds=(w,), env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, bufsize=0,
    )
    os.close(w)
    r_pipe = r

    # Wait for daemon ready (reads stdout until we see "[daemon] ready")
    print("[daemon] waiting for ready...", flush=True)
    while True:
        line = daemon_proc.stdout.readline().decode(errors="replace").strip()
        if not line:
            if daemon_proc.poll() is not None:
                print("[daemon] DIED during startup", flush=True)
                sys.exit(1)
            continue
        print(f"[daemon] {line}", flush=True)
        if "ready" in line.lower() or "listening" in line.lower():
            break
        if "error" in line.lower() or "failed" in line.lower():
            print(f"[daemon] FAILED: {line}", flush=True)
            sys.exit(1)


def write_cmd(cmd_line: str):
    """Send command to daemon stdin."""
    if daemon_proc.poll() is not None:
        raise RuntimeError("daemon exited")
    daemon_proc.stdin.write(cmd_line.encode("utf-8"))
    daemon_proc.stdin.flush()


def read_tokens(n_gen: int) -> list[int]:
    """Read generated tokens from pipe."""
    tokens = []
    for _ in range(n_gen + 50):  # safety margin
        b = os.read(r_pipe, 4)
        if not b or len(b) < 4:
            break
        tok_id = struct.unpack("<i", b)[0]
        if tok_id == -1:
            break
        if tok_id in stop_ids:
            break
        tokens.append(tok_id)
        if len(tokens) >= n_gen:
            break
    return tokens


async def aread_tokens(n_gen: int):
    """Async generator that yields tokens as they arrive."""
    loop = asyncio.get_running_loop()
    count = 0
    while count < n_gen:
        b = await loop.run_in_executor(None, os.read, r_pipe, 4)
        if not b or len(b) < 4:
            break
        tok_id = struct.unpack("<i", b)[0]
        if tok_id == -1:
            break
        if tok_id in stop_ids:
            break
        count += 1
        yield tok_id


@app.get("/health")
async def health():
    if daemon_proc and daemon_proc.poll() is None:
        return {"status": "ok"}
    return JSONResponse({"status": "error", "detail": "daemon dead"}, status_code=503)


@app.get("/v1/models")
async def models():
    return {
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
        "object": "list",
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    tok = get_tokenizer()
    text = apply_chat_template(req.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        prompt_path = f.name
        ids = tokenize_to_file(text, prompt_path)

    gen_len = min(req.max_tokens, 2048)

    async with daemon_lock:
        cmd_line = f"{prompt_path} {gen_len}\n"
        write_cmd(cmd_line)

        if req.stream:
            async def sse():
                async for tok_id in aread_tokens(gen_len):
                    token_text = tok.decode([tok_id])
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                done = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"
                try:
                    os.unlink(prompt_path)
                except Exception:
                    pass

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        t0 = time.perf_counter()
        out_ids = await asyncio.to_thread(read_tokens, gen_len)
        elapsed = time.perf_counter() - t0

    try:
        os.unlink(prompt_path)
    except Exception:
        pass

    content = tok.decode(out_ids, skip_special_tokens=True)
    # Strip think blocks
    if "</think>" in content:
        _, _, content = content.partition("</think>")
        content = content.strip()

    tps = len(out_ids) / elapsed if elapsed > 0 else 0

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop" if len(out_ids) < gen_len else "length",
        }],
        "usage": {
            "prompt_tokens": len(ids),
            "completion_tokens": len(out_ids),
            "total_tokens": len(ids) + len(out_ids),
        },
        "_meta": {"tok_per_sec": round(tps, 1)},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFlash 24GB server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--target", default=str(DFLASH_DIR / "models" / "Qwen3.6-27B-Q4_K_M.gguf"))
    parser.add_argument("--draft", default=str(DFLASH_DIR / "models" / "draft"))
    parser.add_argument("--bin", default=str(DFLASH_DIR / "build" / "test_dflash"))
    parser.add_argument("--budget", type=int, default=6,
                        help="DDTree budget. Lower = less VRAM. 6 fits 24GB, 22 needs 48GB")
    parser.add_argument("--max-ctx", type=int, default=8192,
                        help="Max context. Keep small on 24GB to leave room for rollback cache")
    args = parser.parse_args()

    # Pre-load tokenizer
    get_tokenizer()

    # Start daemon
    start_daemon(args.target, args.draft, args.bin, args.budget, args.max_ctx)

    print(f"[server] DFlash 24GB server on :{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
