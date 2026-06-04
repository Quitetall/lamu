#!/usr/bin/env python3
"""Append one record to the agent-viewer stream (agent-stream.jsonl).
Usage:
  python clog.py --source mimo --label "bible review" --role response --text "..."
  python clog.py --source codex --file /tmp/codex_out.md
  some-cmd | python clog.py --source deepseek --label "audit"
"""
import argparse, json, pathlib, sys, time
STREAM = pathlib.Path.home() / ".local/state/lamu/agent-stream.jsonl"
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="mimo|deepseek|codex|claude|subagent|...")
ap.add_argument("--label", default="")
ap.add_argument("--role", default="response", choices=["prompt", "response"])
ap.add_argument("--text", default=None)
ap.add_argument("--file", default=None)
a = ap.parse_args()
text = (a.text if a.text is not None
        else pathlib.Path(a.file).read_text(errors="replace") if a.file
        else sys.stdin.read())
STREAM.parent.mkdir(parents=True, exist_ok=True)
with STREAM.open("a") as f:
    f.write(json.dumps({"ts": time.time(), "source": a.source, "label": a.label,
                        "role": a.role, "text": text}) + "\n")
print(f"logged {len(text)} chars from {a.source} → {STREAM}")
