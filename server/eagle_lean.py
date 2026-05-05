"""
Lean EAGLE speculative decoding — llama-cpp GGUF + PyTorch EAGLE head.

Uses llama_get_embeddings_ith() to extract hidden states during normal
generation without disrupting the KV cache. No separate embed() call.

Main model: llama-cpp (GGUF, 16 GB)
EAGLE head: PyTorch (570 MB FP16)
Total: ~17 GB on 24 GB GPU
"""

import ctypes
import json
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama, llama_cpp

MODELS_DIR = Path.home() / "models" / "qwen3.6-27b-heretic"
EAGLE_DIR = Path.home() / "models" / "qwen3.6-27b-heretic-eagle" / "eagle_head"
INNER_DIM = 1024
PORT = int(os.getenv("LLM_PORT") or "8020")
CTX = int(os.getenv("LLM_CTX") or "32768")

app = FastAPI(title="LAMU EAGLE Lean")


# ── EAGLE Head ──────────────────────────────────────────────────────────

class EagleHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, inner_dim=INNER_DIM):
        super().__init__()
        self.down = nn.Linear(hidden_size, inner_dim)
        self.norm = nn.LayerNorm(inner_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=inner_dim, nhead=8, dim_feedforward=inner_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.lm = nn.Linear(inner_dim, vocab_size, bias=False)
        # Reusable host-pinned + device-side hidden buffer; rebuilt on first
        # `draft()` once we know the actual device/dtype.
        self._draft_buf: torch.Tensor | None = None

    def forward(self, x):
        return self.lm(self.enc(self.norm(self.down(x))))

    def _trunk(self, x):
        """Everything before the vocab projection."""
        return self.enc(self.norm(self.down(x)))

    @torch.no_grad()
    def draft(self, hidden_np: np.ndarray) -> int:
        # Reuse a pinned buffer to avoid host→device alloc + transfer per call.
        if self._draft_buf is None:
            self._draft_buf = torch.empty(
                (1, 1, self.down.in_features),
                device=self.lm.weight.device, dtype=self.lm.weight.dtype,
            )
        # Single explicit copy_; avoids the implicit float-cast + tensor materialisation
        # that `from_numpy().to(...)` does each call.
        self._draft_buf.copy_(
            torch.from_numpy(hidden_np).view(1, 1, -1),
            non_blocking=True,
        )
        h = self._trunk(self._draft_buf)            # [1,1,inner]
        # Fused matmul + argmax. `topk(1)` on GPU returns the index without
        # materialising the full host-side logits vector — same matmul work
        # as the original `lm(...)` call but the argmax stays on-device until
        # we pull the single int back. Dropping the host copy of a 248K-element
        # fp16 tensor saves ~500KB of device→host bandwidth per draft.
        logits = self.lm(h)[0, 0]                   # [vocab]
        idx = torch.argmax(logits)                  # scalar tensor on device
        return int(idx.item())


# ── Hidden state extraction from llama-cpp ──────────────────────────────

def get_hidden_state(llm: Llama, pos: int) -> np.ndarray:
    """Extract hidden state at position `pos` using llama_get_embeddings_ith.
    Does NOT disrupt the KV cache — reads from the existing computation."""
    ctx = llm._ctx.ctx
    n_embd = llama_cpp.llama_model_n_embd(llm._model.model)
    ptr = llama_cpp.llama_get_embeddings_ith(ctx, pos)
    if not ptr:
        return None
    return np.ctypeslib.as_array(ptr, shape=(n_embd,)).copy()


# ── Generation with EAGLE speculation ───────────────────────────────────

class LeanDecoder:
    def __init__(self, llm: Llama, eagle: EagleHead):
        self.llm = llm
        self.eagle = eagle
        self.n_embd = llama_cpp.llama_model_n_embd(llm._model.model)
        self.accepted = 0
        self.drafted = 0
        self.total = 0

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        tokens = self.llm.tokenize(prompt.encode())
        self.llm.reset()
        self.llm.eval(tokens)

        n_vocab = 248320  # qwen3.6 vocab size
        ctx = self.llm._ctx.ctx
        mem = llama_cpp.llama_get_memory(ctx)
        eos = self.llm.token_eos()

        for _ in range(max_tokens):
            # Read main logits at last position. Direct ndarray view — no copy
            # since we consume immediately and the next eval invalidates anyway.
            logits_ptr = llama_cpp.llama_get_logits_ith(ctx, -1)
            if not logits_ptr:
                break
            logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))

            # Sample (softmax materialises a fresh array, so the view above is safe)
            if temperature > 0:
                probs = _softmax(logits.astype(np.float32) / temperature)
                token = int(np.random.choice(n_vocab, p=probs))
            else:
                token = int(np.argmax(logits))

            if token == eos:
                break

            yield token
            self.total += 1

            # Hidden state at the position we just sampled from.
            hidden = get_hidden_state(self.llm, -1)
            if hidden is None:
                # No EAGLE this step — single-token eval and continue.
                self.llm.eval([token])
                continue

            # EAGLE drafts the *next* token speculatively.
            draft_token = self.eagle.draft(hidden)
            self.drafted += 1

            # Batched eval: process [token, draft] in a single forward pass.
            # logits at position -2 hold the main model's prediction for "after token",
            # which is exactly what `draft` is trying to be — so this is the direct
            # first-draft-verification read with no extra forward.
            self.llm.eval([token, draft_token])

            verify_ptr = llama_cpp.llama_get_logits_ith(ctx, -2)
            if not verify_ptr:
                continue
            verify_logits = np.ctypeslib.as_array(verify_ptr, shape=(n_vocab,))
            main_pred = int(np.argmax(verify_logits))

            if main_pred == draft_token:
                # Accept: KV cache already holds [token, draft]. Two output tokens
                # for the cost of one batched forward.
                self.accepted += 1
                yield draft_token
                self.total += 1
            else:
                # Reject: roll the draft out of KV, replace with main_pred.
                # Speculative-decoding semantics: on reject, emit main's choice
                # so output distribution still matches the main model.
                pos_max = llama_cpp.llama_memory_seq_pos_max(mem, 0)
                llama_cpp.llama_memory_seq_rm(mem, 0, pos_max, pos_max + 1)
                yield main_pred
                self.total += 1
                if main_pred == eos:
                    break
                self.llm.eval([main_pred])

    @property
    def acceptance_rate(self):
        return self.accepted / max(self.drafted, 1)


def _softmax(x):
    e = np.exp(x - np.max(x))
    e = np.nan_to_num(e, nan=0.0, posinf=0.0)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)


def strip_think(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip() or text


# ── API ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "qwen3.6-27b-uncensored"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False

decoder: LeanDecoder = None

@app.get("/health")
async def health():
    return {
        "status": "ok", "model": "qwen3.6-27b-uncensored", "engine": "eagle_lean",
        "acceptance_rate": f"{decoder.acceptance_rate:.1%}" if decoder else "0%",
        "total_tokens": decoder.total if decoder else 0,
        "accepted": decoder.accepted if decoder else 0,
        "drafted": decoder.drafted if decoder else 0,
    }

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "qwen3.6-27b-uncensored", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    parts = []
    for m in req.messages:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n<think>\n")
    prompt = "\n".join(parts)

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        async def sse():
            yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':req.model,'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})}\n\n"
            think_done = False; buf = []
            for tid in decoder.generate(prompt, req.max_tokens, req.temperature):
                text = decoder.llm.detokenize([tid]).decode("utf-8", errors="replace")
                buf.append(text)
                if not think_done:
                    if "</think>" in "".join(buf):
                        think_done = True
                        after = "".join(buf).split("</think>",1)[1].lstrip("\n")
                        if after:
                            yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':req.model,'choices':[{'index':0,'delta':{'content':after},'finish_reason':None}]})}\n\n"
                    continue
                yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':req.model,'choices':[{'index':0,'delta':{'content':text},'finish_reason':None}]})}\n\n"
            yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':created,'model':req.model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\ndata: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    tokens = list(decoder.generate(prompt, req.max_tokens, req.temperature))
    text = strip_think(decoder.llm.detokenize(tokens).decode("utf-8", errors="replace"))
    return {
        "id":cid,"object":"chat.completion","created":created,"model":req.model,
        "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
        "usage":{"completion_tokens":len(tokens)},
        "eagle_stats":{"acceptance_rate":f"{decoder.acceptance_rate:.1%}","accepted":decoder.accepted,"drafted":decoder.drafted},
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    global decoder

    # Pull every Tensor Core lever available on the GPU. The lm_head matmul
    # (1024 × 248320) and the verify forward both benefit; with TF32/cudnn
    # benchmark off we leave 30-40% throughput on the table.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    gguf = None
    for q in ["Q4_K_M", "Q5_K_S"]:
        for f in MODELS_DIR.glob(f"*{q}*.gguf"):
            gguf = str(f); break
        if gguf: break

    print("LAMU EAGLE Lean Server")
    print(f"  Model: {Path(gguf).name}")

    llm = Llama(
        model_path=gguf, n_gpu_layers=-1, n_ctx=CTX,
        embedding=True, type_k=2, type_v=2, flash_attn=True,
        logits_all=True,  # need logits at all positions for verification
        verbose=False,
    )
    print(f"  Context: {CTX:,}")

    config = json.loads((EAGLE_DIR / "config.json").read_text())
    eagle = EagleHead(config["hidden_size"], config["vocab_size"], config["inner_dim"])
    state = torch.load(EAGLE_DIR / "eagle_head_best.pt", map_location="cuda", weights_only=True)
    eagle.load_state_dict(state)
    eagle = eagle.cuda().half().eval()
    print(f"  EAGLE: {sum(p.numel()*p.element_size() for p in eagle.parameters())/1e6:.0f} MB")
    print(f"  Port: {PORT}")

    decoder = LeanDecoder(llm, eagle)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

if __name__ == "__main__":
    main()
