#!/usr/bin/env python3
"""Stream chat with bee server. Toggle thinking via env or flag.

Usage:
    chat-bee.py "Write a haiku about robots"
    chat-bee.py --no-think "What is 2+2?"
    chat-bee.py --port 8021 --temp 0.7 "Tell me a joke"

Env:
    BEE_PORT (default 8021)
    BEE_THINK=0  to disable reasoning by default
"""
import argparse
import json
import os
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream chat to bee server")
    parser.add_argument("prompt", nargs="+", help="prompt text")
    parser.add_argument("--port", default=os.environ.get("BEE_PORT", "8021"))
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--no-think", action="store_true", help="disable Qwen reasoning")
    parser.add_argument("--think", action="store_true", help="force enable thinking")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    # Default think on; --no-think disables; --think wins if both passed.
    enable_thinking = True
    if os.environ.get("BEE_THINK", "1") == "0":
        enable_thinking = False
    if args.no_think:
        enable_thinking = False
    if args.think:
        enable_thinking = True

    prompt = " ".join(args.prompt)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": "any",
        "messages": messages,
        "temperature": args.temp,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens

    url = f"http://localhost:{args.port}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    in_reasoning = False
    have_visible = False
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                if not raw.startswith(b"data: "):
                    continue
                chunk = raw[6:].strip()
                if not chunk or chunk == b"[DONE]":
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                # Reasoning stream (Qwen3.6 <think> content)
                rc = delta.get("reasoning_content")
                if rc:
                    if not in_reasoning:
                        sys.stdout.write("\033[90m[thinking] ")
                        in_reasoning = True
                    sys.stdout.write(rc)
                    sys.stdout.flush()
                # Visible content
                ct = delta.get("content")
                if ct:
                    if in_reasoning:
                        sys.stdout.write("\033[0m\n")
                        in_reasoning = False
                    sys.stdout.write(ct)
                    sys.stdout.flush()
                    have_visible = True
    except KeyboardInterrupt:
        sys.stdout.write("\033[0m\n[interrupted]\n")
        return 130

    if in_reasoning:
        sys.stdout.write("\033[0m")
    if have_visible:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
