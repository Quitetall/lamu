"""Reasoning extractor — per-model-family think-block detection and stripping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from lamu.core.errors import ReasoningOverflow
from lamu.core.types import ReasoningMarker, StreamChunk


# Hard cap on the streaming buffer while inside a think-block. If a backend
# emits more than this without a close-tag, treat it as a malformed stream
# and surface ReasoningOverflow rather than silently growing memory.
_REASONING_BUFFER_CAP_BYTES: int = 64 * 1024


class ReasoningExtractor:
    """Handles think-block detection, stripping, and structured extraction.

    Registered per model family via ReasoningMarker from the model registry.
    """

    def __init__(self, marker: ReasoningMarker) -> None:
        self._marker = marker

    @property
    def marker(self) -> ReasoningMarker:
        return self._marker

    def split(self, text: str) -> tuple[str, str]:
        """Split full response into (reasoning, content).

        Returns ("", text) if no reasoning markers found.
        Returns (reasoning, content) if markers found.
        """
        open_tag = self._marker.open_tag
        close_tag = self._marker.close_tag

        open_idx = text.find(open_tag)
        if open_idx == -1:
            return ("", text)

        close_idx = text.find(close_tag, open_idx + len(open_tag))
        if close_idx == -1:
            # Open tag found but no close — all text after open is reasoning (truncated)
            reasoning = text[open_idx + len(open_tag):]
            return (reasoning.strip(), "")

        reasoning = text[open_idx + len(open_tag):close_idx]
        content = text[close_idx + len(close_tag):]
        return (reasoning.strip(), content.strip())

    def strip(self, text: str) -> str:
        """Strip reasoning, return only content."""
        _, content = self.split(text)
        return content

    def stream_filter(
        self,
        tokens: Iterator[str],
        include_reasoning: bool = False,
    ) -> Iterator[StreamChunk]:
        """Filter a stream of tokens, handling think-block detection.

        Behavior:
        - Buffers tokens until close_tag is detected
        - If include_reasoning=False: suppresses all reasoning tokens,
          then yields content tokens as they arrive
        - If include_reasoning=True: yields reasoning chunks (type="reasoning")
          during think phase, then content chunks (type="content") after

        Handles partial tag matches across token boundaries.
        """
        open_tag = self._marker.open_tag
        close_tag = self._marker.close_tag

        buffer = ""
        in_reasoning = False
        reasoning_done = False

        for token in tokens:
            buffer += token
            if len(buffer) > _REASONING_BUFFER_CAP_BYTES:
                raise ReasoningOverflow(
                    f"reasoning buffer exceeded {_REASONING_BUFFER_CAP_BYTES} bytes"
                    f" without close tag {close_tag!r}"
                )

            if not in_reasoning and not reasoning_done:
                # Haven't entered reasoning yet — check for open tag
                if open_tag in buffer:
                    # Entered reasoning phase
                    in_reasoning = True
                    # Anything before the open tag is pre-content (rare)
                    pre = buffer[:buffer.index(open_tag)]
                    if pre.strip():
                        yield StreamChunk(type="content", text=pre)
                    buffer = buffer[buffer.index(open_tag) + len(open_tag):]
                elif len(buffer) > len(open_tag) * 2:
                    # No open tag found and buffer is long enough — this is content
                    reasoning_done = True
                    yield StreamChunk(type="content", text=buffer)
                    buffer = ""

            elif in_reasoning and not reasoning_done:
                # Inside reasoning — look for close tag
                if close_tag in buffer:
                    # Reasoning complete
                    reasoning_text = buffer[:buffer.index(close_tag)]
                    if include_reasoning and reasoning_text.strip():
                        yield StreamChunk(type="reasoning", text=reasoning_text)
                    buffer = buffer[buffer.index(close_tag) + len(close_tag):]
                    in_reasoning = False
                    reasoning_done = True
                    # Emit any content after close tag
                    if buffer.strip():
                        yield StreamChunk(type="content", text=buffer)
                        buffer = ""
                elif include_reasoning and len(buffer) > 100:
                    # Flush reasoning chunks periodically during long think phase
                    yield StreamChunk(type="reasoning", text=buffer)
                    buffer = ""

            elif reasoning_done:
                # Post-reasoning — everything is content
                yield StreamChunk(type="content", text=token)
                buffer = ""

        # Flush remaining buffer
        if buffer.strip():
            if in_reasoning and not reasoning_done:
                # Never found close tag — treat as truncated reasoning
                if include_reasoning:
                    yield StreamChunk(type="reasoning", text=buffer)
                # No content to emit
            else:
                yield StreamChunk(type="content", text=buffer)


# Null extractor for models without reasoning
class NullReasoningExtractor:
    """For models that don't use think-blocks. Passes through everything."""

    def split(self, text: str) -> tuple[str, str]:
        return ("", text)

    def strip(self, text: str) -> str:
        return text

    def stream_filter(
        self,
        tokens: Iterator[str],
        include_reasoning: bool = False,
    ) -> Iterator[StreamChunk]:
        for token in tokens:
            yield StreamChunk(type="content", text=token)


def get_extractor(marker: Optional[ReasoningMarker]) -> ReasoningExtractor | NullReasoningExtractor:
    """Factory: returns appropriate extractor based on model's marker config."""
    if marker is None:
        return NullReasoningExtractor()
    return ReasoningExtractor(marker)
