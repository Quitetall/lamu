"""
Training pipeline — collect swarm data + QLoRA fine-tune local models.

Feedback loop:
  1. Swarm runs → successful (task, implementation) pairs saved to training_data/
  2. `trainer prepare` → converts pairs to chat-format JSONL
  3. `trainer train` → QLoRA fine-tuning with unsloth on local GPU
  4. `trainer export` → merge LoRA + convert to GGUF (DFlash) or HF (vLLM)
  5. Hot-swap the serving model → local workers get better over time

Usage:
  python -m agents.trainer status
  python -m agents.trainer prepare
  python -m agents.trainer train [--model Qwen/Qwen3.5-27B] [--epochs 3]
  python -m agents.trainer export [--format gguf|hf] [--quant q4_k_m]
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "training_data"
DATASET_DIR = Path(__file__).parent / "datasets"
ADAPTER_DIR = Path(__file__).parent / "adapters"
MERGED_DIR = Path(__file__).parent / "merged_models"


# ── Data collection ─────────────────────────────────────────────────────
# Called automatically by swarm.py on successful runs.
# Can also be called manually to add custom training pairs.

def collect(task: str, plan: list, applied_files: dict, test_output: str = ""):
    """Save a successful (task → implementation) pair for fine-tuning."""
    DATA_DIR.mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "plan": plan,
        "applied_files": applied_files,
        "test_output": test_output,
    }

    path = DATA_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(entry, indent=2))
    print(f"Saved training pair: {path.name}")


# ── Status ──────────────────────────────────────────────────────────────

def status():
    """Show training data and adapter stats."""
    DATA_DIR.mkdir(exist_ok=True)
    ADAPTER_DIR.mkdir(exist_ok=True)

    pairs = sorted(DATA_DIR.glob("*.json"))
    adapters = sorted(ADAPTER_DIR.glob("lora_*"))

    print(f"Training pairs: {len(pairs)}")
    if pairs:
        oldest = pairs[0].name
        newest = pairs[-1].name
        total_files = 0
        for f in pairs:
            data = json.loads(f.read_text())
            total_files += len(data.get("applied_files", {}))
        print(f"  Range: {oldest} → {newest}")
        print(f"  Total file examples: {total_files}")

    print(f"\nLoRA adapters: {len(adapters)}")
    for a in adapters:
        size_mb = sum(f.stat().st_size for f in a.rglob("*") if f.is_file()) / 1e6
        print(f"  {a.name} ({size_mb:.0f} MB)")

    datasets = sorted(DATASET_DIR.glob("*.jsonl")) if DATASET_DIR.exists() else []
    if datasets:
        print(f"\nPrepared datasets: {len(datasets)}")
        for d in datasets:
            lines = sum(1 for _ in open(d))
            print(f"  {d.name} ({lines} examples)")


# ── Dataset preparation ─────────────────────────────────────────────────

def prepare() -> list[dict]:
    """Convert collected swarm data to chat-format JSONL for fine-tuning."""
    DATA_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)

    entries = []
    for f in sorted(DATA_DIR.glob("*.json")):
        data = json.loads(f.read_text())

        instruction = f"Task: {data['task']}"
        if data.get("plan"):
            instruction += f"\n\nPlan:\n{json.dumps(data['plan'], indent=2)}"

        # Build response: complete file contents
        response_parts = []
        for path, content in data.get("applied_files", {}).items():
            response_parts.append(f"```{path}\n{content}\n```")
        response = "\n\n".join(response_parts)

        if not response:
            continue

        entries.append({
            "messages": [
                {"role": "system", "content": "You are an expert programmer. Implement exactly what is described. Output complete file contents in fenced code blocks with the file path as the language tag."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
        })

    output = DATASET_DIR / "swarm_pairs.jsonl"
    with open(output, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Prepared {len(entries)} training examples → {output}")
    return entries


# ── QLoRA fine-tuning ───────────────────────────────────────────────────

def train(
    base_model: str = "Qwen/Qwen3.5-27B",
    lora_r: int = 16,
    lora_alpha: int = 16,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 4096,
    warmup_ratio: float = 0.03,
):
    """Run QLoRA fine-tuning with unsloth (optimized for consumer GPUs)."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("unsloth not installed. Install with:")
        print("  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        print("  pip install --no-deps trl peft accelerate bitsandbytes")
        return None

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    dataset_path = DATASET_DIR / "swarm_pairs.jsonl"
    if not dataset_path.exists():
        print("No dataset found. Run `python -m agents.trainer prepare` first.")
        return None

    n_examples = sum(1 for _ in open(dataset_path))
    print(f"Dataset: {n_examples} examples")
    if n_examples < 5:
        print("Warning: very small dataset. Results may be poor. Collect more swarm runs first.")

    # Load base model in 4-bit
    print(f"Loading base model: {base_model} (4-bit)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # Apply LoRA
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load and format dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def format_chat(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_chat, batched=True, remove_columns=dataset.column_names)

    # Training config
    ADAPTER_DIR.mkdir(exist_ok=True)
    run_name = f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = ADAPTER_DIR / run_name

    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=1,
        save_strategy="epoch",
        fp16=not os.getenv("BF16"),
        bf16=bool(os.getenv("BF16")),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )

    print(f"\nStarting training: {epochs} epochs, lr={learning_rate}, batch={batch_size}x{gradient_accumulation}")
    stats = trainer.train()
    trainer.save_model(str(output_dir))

    print(f"\nTraining complete:")
    print(f"  Loss: {stats.training_loss:.4f}")
    print(f"  Adapter: {output_dir}")
    return output_dir


# ── Export ───────────────────────────────────────────────────────────────

def export(
    adapter_path: str = None,
    format: str = "gguf",
    quant: str = "q4_k_m",
):
    """Merge LoRA + export to GGUF (for DFlash) or HF format (for vLLM)."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("unsloth not installed.")
        return None

    if adapter_path is None:
        adapters = sorted(ADAPTER_DIR.glob("lora_*"))
        if not adapters:
            print("No adapters found. Run training first.")
            return None
        adapter_path = str(adapters[-1])
        print(f"Using latest adapter: {adapter_path}")

    MERGED_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading adapter: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    if format == "gguf":
        out = MERGED_DIR / f"gguf_{ts}"
        print(f"Merging + quantizing to GGUF ({quant})...")
        model.save_pretrained_gguf(str(out), tokenizer, quantization_method=quant)
        gguf_files = list(out.glob("*.gguf"))
        if gguf_files:
            print(f"GGUF ready: {gguf_files[0]}")
            print(f"To serve with DFlash, update scripts/serve-dflash.sh --target to point here.")
        return out
    else:
        out = MERGED_DIR / f"hf_{ts}"
        print("Merging to HF format (for vLLM serving)...")
        model.save_pretrained_merged(str(out), tokenizer)
        print(f"Merged model: {out}")
        print(f"To serve with vLLM, point the docker compose model path here.")
        return out


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Training pipeline for local model fine-tuning")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show training data and adapter stats")
    sub.add_parser("prepare", help="Convert collected data to training format")

    train_p = sub.add_parser("train", help="Run QLoRA fine-tuning")
    train_p.add_argument("--model", default="Qwen/Qwen3.5-27B", help="Base model")
    train_p.add_argument("--epochs", type=int, default=3)
    train_p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    train_p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    train_p.add_argument("--batch", type=int, default=1, help="Batch size")
    train_p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    train_p.add_argument("--max-seq", type=int, default=4096, help="Max sequence length")

    export_p = sub.add_parser("export", help="Merge LoRA + convert to serving format")
    export_p.add_argument("--adapter", default=None, help="Adapter path (default: latest)")
    export_p.add_argument("--format", choices=["gguf", "hf"], default="gguf",
                          help="Output format: gguf (DFlash) or hf (vLLM)")
    export_p.add_argument("--quant", default="q4_k_m", help="GGUF quantization method")

    args = parser.parse_args()

    if args.command == "status":
        status()
    elif args.command == "prepare":
        prepare()
    elif args.command == "train":
        train(
            base_model=args.model,
            epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            batch_size=args.batch,
            gradient_accumulation=args.grad_accum,
            max_seq_length=args.max_seq,
        )
    elif args.command == "export":
        export(adapter_path=args.adapter, format=args.format, quant=args.quant)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
