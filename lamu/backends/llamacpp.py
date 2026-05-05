"""llama.cpp backend — manages llama-server subprocess."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterator, Optional

from lamu.backends.base import Backend
from lamu.core.types import ModelEntry


# Default llama-server binary location
_DEFAULT_BIN = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"


class LlamaCppBackend(Backend):
    """Backend for llama-server (llama.cpp HTTP server)."""

    def __init__(self, bin_path: Optional[Path] = None) -> None:
        self._bin = bin_path or _DEFAULT_BIN
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._port: int = 0
        self._model_name: str = ""
        self._entry: Optional[ModelEntry] = None

    def load(self, entry: ModelEntry, port: int) -> int:
        """Start llama-server with the model."""
        self._entry = entry
        self._port = port
        self._model_name = entry.name

        if not self._bin.exists():
            raise RuntimeError(f"llama-server not found at {self._bin}")

        cmd: list[str] = [
            str(self._bin),
            "-m", str(entry.path),
            "--host", "0.0.0.0",
            "--port", str(port),
            "--ctx-size", str(min(entry.context_max, 131072)),
            "-ngl", "99",
            "--flash-attn", "on",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
            "--parallel", "1",
        ]

        # Add ngram-mod speculation for qwen models
        if entry.arch in ("qwen35", "qwen3"):
            cmd.extend([
                "--spec-type", "ngram-mod",
                "--spec-ngram-mod-n-match", "24",
                "--spec-ngram-mod-n-min", "12",
                "--spec-ngram-mod-n-max", "48",
            ])

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}

        self._proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL,
            stderr=open("/tmp/lamu-llamacpp.log", "w"),
        )

        # Wait for health
        for _ in range(60):
            time.sleep(1)
            if self.is_healthy():
                return self._proc.pid

        # Timeout
        self.unload()
        raise RuntimeError(f"llama-server failed to start within 60s (port {port})")

    def unload(self) -> None:
        if self._proc:
            try:
                self._proc.send_signal(signal.SIGKILL)
                self._proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            self._proc = None
        self._model_name = ""

    def is_healthy(self) -> bool:
        try:
            req = urllib.request.Request(f"http://localhost:{self._port}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 16384,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        req = urllib.request.Request(
            f"http://localhost:{self._port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            # Return content + reasoning_content combined for extractor
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n{content}"
            return content

    def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 16384,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        req = urllib.request.Request(
            f"http://localhost:{self._port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def get_vram_mb(self) -> int:
        if not self._proc:
            return 0
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) == 2 and int(parts[0].strip()) == self._proc.pid:
                    return int(parts[1].strip())
        except (subprocess.TimeoutExpired, ValueError):
            pass
        return 0

    @property
    def port(self) -> int:
        return self._port

    @property
    def model_name(self) -> str:
        return self._model_name
