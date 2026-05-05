"""Tests for server.rag — keyword-based wiki retrieval."""
from __future__ import annotations

import pytest

from server.rag import WikiRAG


@pytest.fixture
def wiki(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "vllm-on-24gb.md").write_text("vLLM hits OOM on 24 GB cards.")
    (pages / "qwen-tuning.md").write_text("Qwen3.6 tuning recipes for max throughput.")
    (pages / "unrelated.md").write_text("totally different topic")
    return WikiRAG(wiki_dir=str(tmp_path))


def test_init_loads_all_pages(wiki):
    assert set(wiki.pages.keys()) == {"vllm-on-24gb", "qwen-tuning", "unrelated"}


def test_list_pages(wiki):
    assert "qwen-tuning" in wiki.list_pages()


def test_retrieve_finds_relevant(wiki):
    out = wiki.retrieve("why doesn't vLLM work on 24GB?")
    assert "vLLM hits OOM" in out


def test_retrieve_returns_empty_when_no_match(wiki):
    assert wiki.retrieve("aaaaaaaaaaaaa zzzzzzzzz") == ""


def test_retrieve_caps_max_pages(wiki):
    out = wiki.retrieve("qwen vllm tuning")
    sections = out.count("--- wiki/")
    assert sections <= 3


def test_missing_wiki_dir_yields_empty(tmp_path):
    rag = WikiRAG(wiki_dir=str(tmp_path / "nope"))
    assert rag.pages == {}
    assert rag.retrieve("anything") == ""
