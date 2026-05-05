"""Tests for lamu.core.reasoning — think-block detection."""
from __future__ import annotations

import pytest

from lamu.core.reasoning import (
    NullReasoningExtractor,
    ReasoningExtractor,
    get_extractor,
)
from lamu.core.types import ReasoningMarker, StreamChunk


@pytest.fixture
def marker():
    return ReasoningMarker(open_tag="<think>", close_tag="</think>", family="qwen35")


@pytest.fixture
def extractor(marker):
    return ReasoningExtractor(marker)


def test_split_no_markers_returns_full_content(extractor):
    assert extractor.split("just a plain answer") == ("", "just a plain answer")


def test_split_extracts_reasoning(extractor):
    text = "<think>step by step</think>final answer"
    reasoning, content = extractor.split(text)
    assert reasoning == "step by step"
    assert content == "final answer"


def test_split_truncated_no_close(extractor):
    text = "<think>still thinking, hit token limit"
    reasoning, content = extractor.split(text)
    assert reasoning == "still thinking, hit token limit"
    assert content == ""


def test_strip_returns_only_content(extractor):
    assert extractor.strip("<think>x</think>y") == "y"


def test_stream_filter_no_reasoning_block(extractor):
    chunks = list(extractor.stream_filter(iter(["hello ", "world"])))
    text = "".join(c.text for c in chunks)
    assert "hello" in text
    assert all(c.type == "content" for c in chunks)


def test_stream_filter_drops_reasoning_by_default(extractor):
    tokens = ["<think>", "stuff ", "more", "</think>", "answer"]
    chunks = list(extractor.stream_filter(iter(tokens), include_reasoning=False))
    content = "".join(c.text for c in chunks if c.type == "content")
    assert "stuff" not in content
    assert "answer" in content


def test_stream_filter_includes_reasoning_when_requested(extractor):
    tokens = ["<think>", "stuff", "</think>", "answer"]
    chunks = list(extractor.stream_filter(iter(tokens), include_reasoning=True))
    types = [c.type for c in chunks]
    assert "reasoning" in types
    assert "content" in types


def test_null_extractor_passthrough():
    e = NullReasoningExtractor()
    assert e.split("hi") == ("", "hi")
    assert e.strip("hi") == "hi"
    chunks = list(e.stream_filter(iter(["a", "b"])))
    assert [c.text for c in chunks] == ["a", "b"]
    assert all(c.type == "content" for c in chunks)


def test_get_extractor_picks_correct_type(marker):
    assert isinstance(get_extractor(marker), ReasoningExtractor)
    assert isinstance(get_extractor(None), NullReasoningExtractor)


def test_stream_filter_buffer_cap_raises(extractor):
    """Phase C: unbounded think buffer raises ReasoningOverflow at 64 KB."""
    from lamu.core.errors import ReasoningOverflow
    huge_token = "x" * 100_000
    with pytest.raises(ReasoningOverflow):
        list(extractor.stream_filter(iter(["<think>", huge_token])))
