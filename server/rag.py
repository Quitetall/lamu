"""
Simple RAG over the LAMU wiki for the local model.

Loads wiki pages, finds relevant ones by keyword matching, and prepends
them as context before the user's prompt. No embeddings needed — the wiki
is small enough for keyword matching to work well.

Usage:
    from server.rag import WikiRAG
    rag = WikiRAG()
    context = rag.retrieve("why doesn't vLLM work on 24GB?")
    # Returns relevant wiki page contents as a string
"""

import os
from pathlib import Path


class WikiRAG:
    """Keyword-based retrieval over the LAMU wiki."""

    def __init__(self, wiki_dir: str = None):
        self.wiki_dir = Path(wiki_dir or os.path.expanduser("~/local-llm/wiki"))
        self.pages = {}
        self._load()

    def _load(self):
        """Load all wiki pages into memory."""
        pages_dir = self.wiki_dir / "pages"
        if not pages_dir.exists():
            return
        for f in pages_dir.glob("*.md"):
            self.pages[f.stem] = f.read_text()

    def retrieve(self, query: str, max_pages: int = 3) -> str:
        """Find relevant wiki pages for a query. Returns concatenated content."""
        if not self.pages:
            return ""

        query_lower = query.lower()
        query_words = set(w for w in query_lower.split() if len(w) > 3)

        # Score each page by keyword overlap
        scored = []
        for name, content in self.pages.items():
            content_lower = content.lower()
            # Score: title match (heavy) + content word matches
            score = 0
            for word in query_words:
                if word in name:
                    score += 10
                score += content_lower.count(word)
            if score > 0:
                scored.append((score, name, content))

        scored.sort(reverse=True)
        top = scored[:max_pages]

        if not top:
            return ""

        parts = []
        for _, name, content in top:
            parts.append(f"--- wiki/{name} ---\n{content}\n")

        return "\n".join(parts)

    def list_pages(self) -> list[str]:
        return list(self.pages.keys())
