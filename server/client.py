"""
Local LLM Python client — import from anywhere.

    from server.client import LocalLLM

    llm = LocalLLM()
    print(llm.models())
    print(llm.chat("explain quicksort"))
    print(llm.chat("implement it in rust", model="dflash/luce-dflash"))

Works with any OpenAI-compatible backend. Defaults to Bifrost gateway.
"""

import json
import os
import urllib.request
import urllib.error
from typing import Iterator, Optional


class LocalLLM:
    """Minimal client for the local LLM stack. Zero dependencies beyond stdlib."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        default_model: str = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("BIFROST_URL")
            or "http://localhost:8080/v1"
        )
        self.api_key = api_key or os.getenv("LLM_API_KEY", "sk-local")
        self.default_model = (
            default_model
            or os.getenv("LLM_MODEL")
            or "qwen/qwen3.6-35b-uncensored"
        )

    def _request(self, path: str, payload: dict = None, timeout: int = 120) -> dict:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    @staticmethod
    def _strip_think(text: str) -> str:
        if "</think>" in text:
            _, _, text = text.partition("</think>")
        return text.strip()

    def models(self) -> list[str]:
        """List available model IDs by probing all known local endpoints."""
        found = []
        endpoints = [
            ("http://localhost:8020/v1", "qwen"),    # Qwen3.6
            ("http://localhost:8000/v1", "dflash"),   # DFlash
            ("http://localhost:8001/v1", "sglang"),   # SGLang
            ("http://localhost:9001/v1", "gpt2"),     # GPT-2 proxy
        ]
        for base, provider in endpoints:
            try:
                req = urllib.request.Request(f"{base}/models")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read())
                    for m in data.get("data", []):
                        found.append(f"{provider}/{m['id']}")
            except Exception:
                pass
        return found

    def chat(
        self,
        prompt: str,
        system: str = "",
        model: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        raw: bool = False,
    ) -> str:
        """Send a chat completion. Returns the response text.

        Args:
            prompt: User message
            system: Optional system prompt
            model: Model name (provider/model format for Bifrost)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            raw: If True, don't strip <think> blocks
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = self._request("/chat/completions", {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        })

        content = data["choices"][0]["message"]["content"]
        return content if raw else self._strip_think(content)

    def chat_multi(
        self,
        messages: list[dict],
        model: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        raw: bool = False,
    ) -> str:
        """Send a multi-turn conversation. Returns the response text."""
        data = self._request("/chat/completions", {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        })

        content = data["choices"][0]["message"]["content"]
        return content if raw else self._strip_think(content)

    def stream(
        self,
        prompt: str,
        system: str = "",
        model: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream tokens from a chat completion."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    def is_running(self) -> bool:
        """Check if the LLM backend is reachable."""
        try:
            self._request("/models", timeout=3)
            return True
        except Exception:
            return False

    def health(self) -> dict:
        """Get detailed health status of all known endpoints."""
        endpoints = {
            "bifrost": "http://localhost:8080",
            "qwen36": "http://localhost:8020",
            "dflash": "http://localhost:8000",
            "sglang": "http://localhost:8001",
            "gpt2proxy": "http://localhost:9001",
        }
        status = {}
        for name, base in endpoints.items():
            try:
                url = f"{base}/health" if name in ("bifrost", "gpt2proxy") else f"{base}/v1/models"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    status[name] = "up"
            except Exception:
                status[name] = "down"
        return status


# Convenience: module-level singleton
_default = None

def get_default() -> LocalLLM:
    global _default
    if _default is None:
        _default = LocalLLM()
    return _default

def chat(prompt: str, **kwargs) -> str:
    """Quick one-liner: `from server.client import chat; print(chat("hi"))`"""
    return get_default().chat(prompt, **kwargs)

def models() -> list[str]:
    return get_default().models()
