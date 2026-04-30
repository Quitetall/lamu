# Self-Improving Training Loop

## The Feedback Loop
```
swarm runs → tests pass + critic approves → (task, implementation) saved
    ↓
accumulate training data → prepare dataset (JSONL)
    ↓
QLoRA fine-tune on local GPU → export to GGUF
    ↓
serve fine-tuned model → swarm workers are better at YOUR code
    ↓
repeat
```

## Data Collection
Automatic. Every successful swarm run calls `_save_training_data()` in `agents/swarm.py`. Saved to `agents/training_data/` as JSON with:
- task description
- plan (subtasks)
- applied_files (complete file contents)
- test_output
- retry_count, planner_loops

## Commands
```
just train-status        # show collected pairs
just train-prepare       # convert to JSONL chat format
just train               # QLoRA fine-tune (unsloth, 4090)
just train-export-gguf   # merge LoRA + convert to GGUF for serving
just train-export-hf     # merge to HF format (for vLLM/SGLang)
```

## Why This Matters
The model improves specifically at YOUR codebase conventions, architecture patterns, and coding style. Generic benchmarks stay the same, but domain-specific tasks get significantly better.

## Benchmark Baseline
Local swarm (all Qwen3.6, no cloud): 4/5 passed (80%).
The 1 failure was missing fastapi dep, not model error.
Effectively 100% on tasks where deps are available.
