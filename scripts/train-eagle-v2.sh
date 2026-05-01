#!/usr/bin/env bash
# scripts/train-eagle-v2.sh — proper EAGLE training with all improvements
#
# 1. 20K samples (70% code + 30% general)
# 2. Shared frozen LM head from base model
# 3. Full hidden size (5120) transformer layers (~30M trainable)
# 4. Multi-token drafting support (autoregressive EAGLE head)
#
# Data gen: ~3 hours (20K samples × 4-bit model)
# Training: ~30 min (tiny trainable params, frozen LM head)
set -euo pipefail

ROOT="$HOME/local-llm"
VENV="$ROOT/.venv"
MODEL_ID="llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
EAGLE_DIR="$HOME/models/qwen3.6-27b-heretic-eagle"
DATA_DIR="$EAGLE_DIR/train_data_v2"
OUTPUT_DIR="$EAGLE_DIR/eagle_head_v2"

echo -e "\n\033[1mEAGLE v2 Training Pipeline\033[0m"
echo -e "  \033[90m20K samples | shared LM head | multi-token draft\033[0m\n"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

# ── Step 1: Generate training data ──────────────────────────────────────
echo "[1/3] Generating training data (20K samples, 70% code + 30% general)..."

"$VENV/bin/python" << 'PYEOF'
import json, os, sys, torch, numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

MODEL_ID = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
DATA_DIR = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle/train_data_v2"))
NUM_CODE = 14000   # 70% code
NUM_GENERAL = 6000 # 30% general
MAX_LEN = 1024

# Resume from existing
existing = len(list(DATA_DIR.glob("sample_*.npz")))
TOTAL = NUM_CODE + NUM_GENERAL
if existing >= TOTAL:
    print(f"  Already have {existing} samples, skipping data gen")
    sys.exit(0)

print(f"  Resuming from {existing} existing samples", flush=True)

print("  Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("  Loading model (4-bit)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"),
    device_map="auto", trust_remote_code=True, dtype=torch.bfloat16, output_hidden_states=True,
)
model.eval()

# Also extract token embeddings (needed for multi-token drafting)
embed_weight = model.get_input_embeddings().weight.data.cpu()
emb_path = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle/token_embeddings.pt"))
if not emb_path.exists():
    torch.save(embed_weight, emb_path)
    print(f"  Saved token embeddings: {embed_weight.shape}", flush=True)

# Load datasets
print("  Loading datasets...", flush=True)
try:
    code_ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
    code_iter = iter(code_ds)
    code_key = "content"
    print("  Code: bigcode/starcoderdata (python)", flush=True)
except:
    code_ds = load_dataset("codeparrot/github-code", languages=["Python"], split="train", streaming=True)
    code_iter = iter(code_ds)
    code_key = "code"
    print("  Code: codeparrot/github-code (Python)", flush=True)

general_ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
print(f"  General: ShareGPT ({len(general_ds)} conversations)", flush=True)

saved = existing
gen_idx = 0

def process_text(text, saved_count):
    """Run model on text, save hidden states."""
    inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    if input_ids.shape[1] < 64:
        return False
    try:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0].float().cpu().numpy().astype(np.float16)
        targets = input_ids[0, 1:].cpu().numpy()
        # Also save the input token IDs (needed for embedding lookup during training)
        input_toks = input_ids[0, :-1].cpu().numpy()
        np.savez_compressed(
            DATA_DIR / f"sample_{saved_count:05d}.npz",
            hidden_states=hidden[:-1],
            target_tokens=targets,
            input_tokens=input_toks,
        )
        del outputs, hidden
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False

# Generate code samples
print(f"  Generating code samples ({NUM_CODE})...", flush=True)
while saved < existing + NUM_CODE:
    try:
        sample = next(code_iter)
        text = sample.get(code_key, "")
        if len(text) < 200:
            continue
        if process_text(text, saved):
            saved += 1
            if saved % 200 == 0:
                print(f"    {saved}/{TOTAL} (code phase)", flush=True)
    except StopIteration:
        break
    except Exception:
        continue

# Generate general samples
print(f"  Generating general samples ({NUM_GENERAL})...", flush=True)
while saved < existing + TOTAL and gen_idx < len(general_ds):
    sample = general_ds[gen_idx]
    gen_idx += 1
    if "conversations" in sample:
        text = " ".join([c.get("value", "") for c in sample["conversations"]])
    elif "text" in sample:
        text = sample["text"]
    else:
        continue
    if len(text) < 200:
        continue
    if process_text(text, saved):
        saved += 1
        if saved % 200 == 0:
            print(f"    {saved}/{TOTAL} (general phase)", flush=True)

print(f"  Data generation complete: {saved} samples", flush=True)
del model
torch.cuda.empty_cache()
PYEOF

echo "  Data generation done."

# ── Step 2: Train EAGLE head with shared LM head ───────────────────────
echo "[2/3] Training EAGLE v2 head (shared LM head, multi-token)..."

"$VENV/bin/python" << 'PYEOF'
import os, torch, torch.nn as nn, numpy as np, json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

MODEL_ID = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2"
EAGLE_DIR = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle"))
DATA_DIR = EAGLE_DIR / "train_data_v2"
OUTPUT_DIR = EAGLE_DIR / "eagle_head_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H = 5120
V = 248320

print(f"H={H} V={V}", flush=True)

# ── Dataset ──────────────────────────────────────────────────────────
class EagleDataset(Dataset):
    def __init__(self, data_dir, max_len=512):
        self.files = sorted(Path(data_dir).glob("sample_*.npz"))
        self.max_len = max_len
        print(f"  {len(self.files)} training files", flush=True)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        ml = self.max_len
        hidden = torch.from_numpy(d["hidden_states"][:ml].astype(np.float32))
        targets = torch.from_numpy(d["target_tokens"][:ml].astype(np.int64))
        input_toks = torch.from_numpy(d["input_tokens"][:ml].astype(np.int64))
        return hidden, targets, input_toks

def collate(batch):
    hs, ts, itoks = zip(*batch)
    ml = max(h.shape[0] for h in hs)
    ph = torch.zeros(len(hs), ml, H)
    pt = torch.full((len(ts), ml), -1, dtype=torch.long)
    pi = torch.zeros(len(itoks), ml, dtype=torch.long)
    for i, (h, t, it) in enumerate(zip(hs, ts, itoks)):
        ph[i, :h.shape[0]] = h
        pt[i, :t.shape[0]] = t
        pi[i, :it.shape[0]] = it
    return ph, pt, pi

# ── EAGLE v2 Head ────────────────────────────────────────────────────
class EagleHeadV2(nn.Module):
    """Proper EAGLE: full hidden size, shared frozen LM head, residual,
    token embedding fusion for multi-token drafting."""

    def __init__(self, hidden_size, vocab_size, lm_head_weight, embed_weight):
        super().__init__()
        # Fusion: hidden_state + token_embedding → hidden_size
        self.fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.norm_in = nn.LayerNorm(hidden_size)

        # 2 transformer layers at FULL hidden size
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=20,  # 5120/20 = 256 head dim
            dim_feedforward=hidden_size * 2,  # smaller FFN to save memory
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm_out = nn.LayerNorm(hidden_size)

        # Frozen LM head from base model
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(lm_head_weight, requires_grad=False)

        # Frozen token embeddings from base model (for multi-token draft)
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.token_embed.weight = nn.Parameter(embed_weight, requires_grad=False)

    def forward(self, hidden_states, input_token_ids):
        """
        hidden_states: [batch, seq, hidden_size] — from main model
        input_token_ids: [batch, seq] — token IDs for embedding lookup
        """
        # Fuse hidden state with token embedding
        tok_emb = self.token_embed(input_token_ids)
        fused = self.fuse(torch.cat([hidden_states, tok_emb], dim=-1))

        # Transformer with residual
        x = self.norm_in(fused)
        x = self.transformer(x) + hidden_states  # residual from hidden states
        x = self.norm_out(x)

        return self.lm_head(x)

    @torch.no_grad()
    def draft_multi(self, hidden_state, last_token_id, n_draft=4):
        """Generate n_draft tokens autoregressively."""
        device = self.lm_head.weight.device
        dtype = self.lm_head.weight.dtype

        h = hidden_state.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)
        t = torch.tensor([[last_token_id]], device=device, dtype=torch.long)

        drafts = []
        for _ in range(n_draft):
            tok_emb = self.token_embed(t)
            fused = self.fuse(torch.cat([h, tok_emb], dim=-1))
            x = self.norm_in(fused)
            x = self.transformer(x) + h
            x = self.norm_out(x)
            logits = self.lm_head(x)
            next_token = logits[0, -1].argmax().item()
            drafts.append(next_token)

            # Use the transformer output as the next hidden state estimate
            h = x
            t = torch.tensor([[next_token]], device=device, dtype=torch.long)

        return drafts

# ── Training ─────────────────────────────────────────────────────────
device = torch.device("cuda")

print("  Loading frozen weights...", flush=True)
lm_head_w = torch.load(EAGLE_DIR / "lm_head.pt", map_location="cpu", weights_only=True)
embed_w = torch.load(EAGLE_DIR / "token_embeddings.pt", map_location="cpu", weights_only=True)
print(f"  LM head: {lm_head_w.shape}, Embeddings: {embed_w.shape}", flush=True)

dataset = EagleDataset(DATA_DIR, max_len=256)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,
                    pin_memory=True, collate_fn=collate)

model = EagleHeadV2(H, V, lm_head_w, embed_w).to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
print(f"  Trainable: {trainable:.0f}M | Frozen: {frozen:.0f}M", flush=True)

import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-5, weight_decay=0.01
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
crit = nn.CrossEntropyLoss(ignore_index=-1)
scaler = torch.amp.GradScaler('cuda')

best_loss = float('inf')
for ep in range(10):
    model.train(); total = 0; n = 0
    for h, t, itoks in loader:
        h, t, itoks = h.to(device), t.to(device), itoks.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(h, itoks)
            loss = crit(logits.view(-1, V), t.view(-1))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update()
        total += loss.item(); n += 1
        if n % 500 == 0:
            print(f"    batch {n}: loss={loss.item():.4f}", flush=True)
    sched.step()
    avg = total / max(n, 1)
    print(f"  Epoch {ep+1}/10: loss={avg:.4f}", flush=True)
    if avg < best_loss:
        best_loss = avg
        # Save only trainable weights
        trainable_state = {k: v for k, v in model.state_dict().items()
                          if "lm_head" not in k and "token_embed" not in k}
        torch.save(trainable_state, OUTPUT_DIR / "eagle_head_v2_best.pt")
        print(f"    -> saved best", flush=True)

# Save final + config
trainable_state = {k: v for k, v in model.state_dict().items()
                  if "lm_head" not in k and "token_embed" not in k}
torch.save(trainable_state, OUTPUT_DIR / "eagle_head_v2.pt")
(OUTPUT_DIR / "config.json").write_text(json.dumps({
    "hidden_size": H, "vocab_size": V, "num_layers": 2, "num_heads": 20,
    "ffn_mult": 2, "base_model": MODEL_ID, "architecture": "eagle_v2_shared_lm_head",
    "hidden_layer": -1, "n_draft": 4, "uses_token_embeddings": True,
    "shared_weights": ["lm_head.pt", "token_embeddings.pt"],
    "training": {"epochs": 10, "best_loss": best_loss}
}, indent=2))
sz = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
print(f"\nDone! Best loss: {best_loss:.4f} | Size: {sz:.0f} MB", flush=True)
PYEOF

echo "[3/3] Training complete."
echo ""
echo -e "\033[32mEAGLE v2 ready!\033[0m"
echo "  Serve with: just serve-eagle"
