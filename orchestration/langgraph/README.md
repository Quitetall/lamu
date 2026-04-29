# LangGraph local-AI orchestration

## Setup

```bash
cp .env.example .env   # fill in LANGFUSE_PUBLIC_KEY / SECRET_KEY
bash setup.sh
source .venv/bin/activate
```

## Run

```bash
python agents/simple.py "hello"
```

## Add agents

Drop a new file in `agents/` and import `llm`, `langfuse_handler`, and `get_config` from `base`.
Follow the `simple.py` pattern: define a `StateGraph`, compile it, add a `__main__` block.

## Models

| Model | Notes |
|---|---|
| `qwen3.5-27b` | Default — smart, use for reasoning tasks |
| `gpt2-xl` | Switch via `DEFAULT_MODEL=gpt2-xl` — chaotic, use for fun |

Override per-run: `DEFAULT_MODEL=gpt2-xl python agents/simple.py "chaos mode"`
