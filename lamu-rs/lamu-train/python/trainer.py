#!/usr/bin/env python3
"""lamu-train trainer subprocess.

Wire protocol: one StatusUpdate JSON per line on stdout, no envelope.
Schema mirrors `lamu-train/src/protocol.rs::StatusUpdate` exactly.

Two modes:

  python trainer.py <spec_json>
    Real run. Imports unsloth + trl lazily so --self-check stays
    importable without the heavy deps.

  python trainer.py --self-check
    Protocol round-trip test: emit one Step + one Done, no GPU work,
    no model load. Used by Rust integration tests to verify
    line buffering, JSON parsing, and exit-code handling without
    requiring the training venv.

Failure path: any uncaught exception is converted to a Failed
StatusUpdate JSON line, then sys.exit(1). Rust treats Failed as a
terminal status.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def emit(update: dict[str, Any]) -> None:
    """Print one StatusUpdate JSON line. Always flush — Rust reader
    blocks on line completion, not on any timeout."""
    print(json.dumps(update), flush=True)


def emit_failed(error: str) -> None:
    emit({"kind": "failed", "error": error})


def self_check() -> int:
    """Protocol smoke. Emits the same line shapes as a real run, in
    the same order — Step events, then Done — so the Rust reader's
    state machine sees realistic input. Useful in CI without a GPU."""
    emit(
        {
            "kind": "step",
            "step": 1,
            "total": 2,
            "loss": 1.234,
            "lr": 0.0002,
            "vram_mb": 0,
        }
    )
    emit(
        {
            "kind": "step",
            "step": 2,
            "total": 2,
            "loss": 0.987,
            "lr": 0.0001,
            "vram_mb": 0,
        }
    )
    emit(
        {
            "kind": "done",
            "final_loss": 0.987,
            "checkpoint_dir": "/tmp/lamu-train-self-check",
        }
    )
    return 0


def parse_method(method: dict[str, Any]) -> tuple[str, int, int]:
    """Map TrainSpec.method (kind-tagged) to (mode, rank, alpha).

    mode is one of 'qlora' | 'lora' | 'full'. rank/alpha are 0 for
    the full path (no LoRA adapters)."""
    kind = method.get("kind")
    if kind == "q_lora":
        return "qlora", int(method["rank"]), int(method["alpha"])
    if kind == "lora":
        return "lora", int(method["rank"]), int(method["alpha"])
    if kind == "full":
        return "full", 0, 0
    raise ValueError(f"unknown method kind: {kind!r}")


def parse_dataset(dataset: dict[str, Any]) -> Path:
    """Resolve TrainSpec.dataset to a JSONL file on disk.

    The Rust caller is expected to materialize Conversations and
    Registered datasets to JsonlPath before launching the trainer —
    this script accepts only JsonlPath at runtime to keep the
    Python side stateless. Other shapes are a programming error."""
    kind = dataset.get("kind")
    if kind == "jsonl_path":
        return Path(dataset["path"])
    raise ValueError(
        f"trainer.py only accepts dataset.kind=jsonl_path at runtime "
        f"(got {kind!r}). Materialize Conversations/Registered upstream."
    )


def transformers_optim(opt: str) -> str:
    return {
        "adam_w": "adamw_torch",
        "adam_w8bit": "adamw_8bit",
        "apollo_mini": "apollo_mini",
        "apollo_rank4": "apollo",
    }[opt]


def run(spec: dict[str, Any]) -> int:
    """Real training path. Lazy-imports the heavy deps so import
    errors land in a Failed status instead of an opaque traceback."""
    try:
        import torch
        from transformers import TrainerCallback, TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        emit_failed(
            f"missing python deps: {e}. "
            f"Install via the lamu-train pyproject.toml: "
            f"pip install unsloth trl peft bitsandbytes transformers accelerate"
        )
        return 1

    mode, rank, alpha = parse_method(spec["method"])
    dataset_path = parse_dataset(spec["dataset"])

    # Defer the dataset module so the import error message above is
    # informative if transformers is the missing one.
    sys.path.insert(0, str(Path(__file__).parent))
    from datasets_loader import load_jsonl_dataset  # noqa: E402

    if not dataset_path.exists():
        emit_failed(f"dataset path does not exist: {dataset_path}")
        return 1

    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    class StatusEmitter(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                vram_mb = (
                    int(torch.cuda.memory_allocated() / (1024 * 1024))
                    if torch.cuda.is_available()
                    else 0
                )
                emit(
                    {
                        "kind": "step",
                        "step": int(state.global_step),
                        "total": int(state.max_steps or state.global_step),
                        "loss": float(logs["loss"]),
                        "lr": float(logs.get("learning_rate", 0.0)),
                        "vram_mb": vram_mb,
                    }
                )

        def on_save(self, args, state, control, **kw):
            emit({"kind": "saved", "path": args.output_dir})

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if metrics and "eval_loss" in metrics:
                emit(
                    {
                        "kind": "eval",
                        "step": int(state.global_step),
                        "eval_loss": float(metrics["eval_loss"]),
                    }
                )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec["base_model"],
        max_seq_length=int(spec["seq_len"]),
        load_in_4bit=mode == "qlora",
    )

    if mode in ("qlora", "lora"):
        model = FastLanguageModel.get_peft_model(
            model,
            r=rank,
            lora_alpha=alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    train_ds = load_jsonl_dataset(
        dataset_path, tokenizer=tokenizer, max_seq_len=int(spec["seq_len"])
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        callbacks=[StatusEmitter()],
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(spec["batch_size"]),
            gradient_accumulation_steps=int(spec["grad_accum"]),
            num_train_epochs=int(spec["epochs"]),
            learning_rate=float(spec["lr"]),
            optim=transformers_optim(spec["optimizer"]),
            seed=int(spec["seed"]),
            logging_steps=1,
            report_to="none",
            save_strategy="epoch",
        ),
    )

    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    emit(
        {
            "kind": "done",
            "final_loss": float(result.training_loss),
            "checkpoint_dir": str(output_dir),
        }
    )
    sys.stderr.write(f"[trainer.py] training finished in {elapsed:.1f}s\n")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        emit_failed("usage: trainer.py <spec_json> | --self-check")
        return 2
    if argv[1] == "--self-check":
        return self_check()
    try:
        spec = json.loads(argv[1])
    except json.JSONDecodeError as e:
        emit_failed(f"spec arg is not valid JSON: {e}")
        return 2
    try:
        return run(spec)
    except Exception as e:
        emit_failed(
            f"{type(e).__name__}: {e}\n"
            f"--- traceback ---\n{traceback.format_exc()}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
