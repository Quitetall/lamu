"""EAGLE v2 training — standalone script, no datasets library."""
import os, torch, torch.nn as nn, numpy as np, json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

EAGLE_DIR = Path(os.path.expanduser("~/models/qwen3.6-27b-heretic-eagle"))
DATA_DIR = EAGLE_DIR / "train_data"
OUTPUT_DIR = EAGLE_DIR / "eagle_head_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
H, V = 5120, 248320

print(f"EAGLE v2 — shared LM head, full hidden size, residual", flush=True)

class DS(Dataset):
    def __init__(self, d, ml=256):
        self.files = sorted(Path(d).glob("sample_*.npz"))
        self.ml = ml
        print(f"  {len(self.files)} files", flush=True)
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        ml = self.ml
        h = torch.from_numpy(d["hidden_states"][:ml].astype(np.float32))
        t = torch.from_numpy(d["target_tokens"][:ml].astype(np.int64))
        n = h.shape[0]
        it = torch.zeros(n, dtype=torch.long)
        toks = d["target_tokens"][:ml]
        if len(toks) > 1:
            it[1:min(len(toks), n)] = torch.from_numpy(toks[:min(len(toks)-1, n-1)].astype(np.int64))
        return h, t[:n], it

def collate(batch):
    hs, ts, itoks = zip(*batch)
    ml = max(h.shape[0] for h in hs)
    ph = torch.zeros(len(hs), ml, H)
    pt = torch.full((len(ts), ml), -1, dtype=torch.long)
    pi = torch.zeros(len(itoks), ml, dtype=torch.long)
    for i, (h, t, it) in enumerate(zip(hs, ts, itoks)):
        ph[i,:h.shape[0]] = h; pt[i,:t.shape[0]] = t; pi[i,:it.shape[0]] = it
    return ph, pt, pi

class EagleV2(nn.Module):
    def __init__(self, lm_w, emb_w):
        super().__init__()
        self.fuse = nn.Linear(H*2, H)
        self.norm_in = nn.LayerNorm(H)
        layer = nn.TransformerEncoderLayer(d_model=H, nhead=20, dim_feedforward=H*2,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm_out = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, V, bias=False)
        self.lm_head.weight = nn.Parameter(lm_w, requires_grad=False)
        self.token_embed = nn.Embedding(V, H)
        self.token_embed.weight = nn.Parameter(emb_w, requires_grad=False)
    def forward(self, hidden, token_ids):
        tok_emb = self.token_embed(token_ids)
        fused = self.fuse(torch.cat([hidden, tok_emb], dim=-1))
        x = self.norm_in(fused)
        x = self.transformer(x) + hidden
        x = self.norm_out(x)
        return self.lm_head(x)

device = torch.device("cuda")
print("  Loading frozen weights...", flush=True)
lm_w = torch.load(EAGLE_DIR / "lm_head.pt", map_location="cpu", weights_only=True)
emb_w = torch.load(EAGLE_DIR / "token_embeddings.pt", map_location="cpu", weights_only=True)
print(f"  LM: {lm_w.shape}, Emb: {emb_w.shape}", flush=True)

loader = DataLoader(DS(DATA_DIR), batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)
model = EagleV2(lm_w, emb_w).to(device)
tr = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e6
print(f"  Trainable: {tr:.0f}M | Frozen: {fr:.0f}M", flush=True)

import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
crit = nn.CrossEntropyLoss(ignore_index=-1)
scaler = torch.amp.GradScaler('cuda')

best = float('inf')
for ep in range(10):
    model.train(); total=0; n=0
    for h, t, itoks in loader:
        h, t, itoks = h.to(device), t.to(device), itoks.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = crit(model(h, itoks).view(-1, V), t.view(-1))
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update()
        total += loss.item(); n += 1
        if n % 200 == 0: print(f"    batch {n}: loss={loss.item():.4f}", flush=True)
    sched.step(); avg = total/max(n,1)
    print(f"  Epoch {ep+1}/10: loss={avg:.4f}", flush=True)
    if avg < best:
        best = avg
        st = {k:v for k,v in model.state_dict().items() if "lm_head" not in k and "token_embed" not in k}
        torch.save(st, OUTPUT_DIR / "eagle_head_v2_best.pt")
        print(f"    -> saved best", flush=True)

(OUTPUT_DIR / "config.json").write_text(json.dumps({
    "hidden_size": H, "vocab_size": V, "num_layers": 2, "num_heads": 20,
    "architecture": "eagle_v2_shared_lm_head", "n_draft": 4, "best_loss": best
}, indent=2))
print(f"\nDone! Best loss: {best:.4f}", flush=True)
