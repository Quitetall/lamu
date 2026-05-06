# Legacy v1.x stack — preserved for reference

This directory holds every launch path from LAMU v1 — the script-driven
"swap a model into :8020, sidecar a small one into :8001, run Bifrost
on :8080" workflow. v1 still works, but it's no longer the recommended
way to use LAMU.

**Use `lamu` (the Rust binary) for new work.** See the project README's
Quick Start.

## What lives here

```
legacy/
├── scripts/
│   ├── start.sh                 # the v1 stack boot — Qwen3.6 + megakernel + Bifrost + Langfuse + Chainlit
│   ├── stop.sh                  # kills everything start.sh launched
│   ├── swap.sh, swap-model.sh   # rotate the model on :8020
│   ├── serve-qwen36.sh          # Qwen3.6 ngram-mod on :8020
│   ├── serve-qwen36-fast.sh     # Qwen3.6 with extra speculative tuning
│   ├── serve-megakernel.sh      # Qwen3.5-0.8B megakernel sidecar on :8001
│   ├── serve-dflash.sh          # DFlash speculative server on :8000 (106 t/s)
│   ├── serve-eagle.sh           # EAGLE speculative experiment
│   ├── serve-vllm.sh            # vLLM backend (research)
│   ├── serve-vllm-qwen36.sh     # ditto, Qwen3.6
│   ├── serve-sglang.sh          # SGLang backend (research)
│   ├── serve-sglang-presets.sh  # SGLang preset matrix
│   ├── serve-comfyui.sh         # ComfyUI image-gen sidecar
│   ├── serve-langfuse.sh        # Langfuse observability stack
│   ├── stop-langfuse.sh         # ...and its shutdown
│   ├── stop-sglang.sh           # SGLang shutdown
│   ├── chat.sh                  # v1 chat shim
│   └── doctor.sh                # v1 diagnostic (v3 doctor lands later)
└── cli/
    └── chat_repl.py             # v1 REPL — talked direct to backends + Bifrost.
                                  # v3 replacement: `lamu repl` (Rust).
```

## Why these are here, not deleted

- 106 t/s DFlash and 494 t/s megakernel numbers were measured against
  these scripts. Reproducing the README's perf table needs them.
- A few of them (`serve-dflash.sh`, `serve-megakernel.sh`) build the
  custom processes that the v3 daemon's `dflash` and `megakernel`
  backends spawn under the hood. Those backends shell out — the
  scripts encode the exact arg sets.
- Some workflows (training, quantization, vLLM/SGLang research) only
  ever existed at the script level. Removing them would lose the
  recipes.

## Why don't I see them in `just`?

The `justfile` was rewritten to put `lamu` (Rust) at the top. v1 targets
moved into a `# ── Legacy v1 ──` section that points at `legacy/scripts/`
paths. They still work — `just swap 3.6` is the same command, just
delegating to `legacy/scripts/swap-model.sh`.

## When can these be deleted?

Once the v3 daemon's `load_model` path covers DFlash + megakernel
end-to-end (currently it does, but only with the legacy launch scripts
present as the spawn target), these can go. Until then, keep.

## What's NOT here

Still in `scripts/` because they're used by the v3 quick-start or are
research-only:

- `setup-*.sh` — model downloaders (`just setup-qwen36`, etc.)
- `quantize-*.sh` — quantization recipes
- `gen_eagle_data.py`, `train-eagle-*.{sh,py}`, `convert_eagle_*.py` — EAGLE research
- `serve-bifrost.sh`, `stop-bifrost.sh` — pending the Bifrost benchmark

Bifrost stays in `scripts/` until the v3 plan's Phase 1 benchmark says
otherwise.
