"""
search.py
=========
Search backend for the evaluation pipeline.
Wraps Tavily into a simple search(query) -> str interface.

Can be extended with other backends (Brave, Exa, etc.) by adding
new classes that implement the same search() method.
"""

import os

# ── Configuration ────────────────────────────────────────────────────────────

TAVILY_MAX_RESULTS = 3      # top-k search results per query
MAX_INFO_CHARS     = 1500   # cap total text per search call


# ═══════════════════════════════════════════════════════════════════════════════
#  Tavily Backend
# ═══════════════════════════════════════════════════════════════════════════════

class TavilySearch:
    """Wraps the Tavily API into a simple search(query) -> str interface."""

    def __init__(self, api_key: str | None = None):
        from tavily import TavilyClient

        key = ""
        #api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise ValueError(
                "Tavily API key required. Pass --tavily-key or set TAVILY_API_KEY."
            )
        self.client = TavilyClient(api_key=key)
        self.total_calls = 0

    def search(self, query: str) -> str:
        """
        Execute a search and return formatted results as a string.
        Format mimics what Search-R1 was trained on: Doc N(Title: "...") content
        """
        self.total_calls += 1
        try:
            response = self.client.search(
                query=query,
                max_results=TAVILY_MAX_RESULTS,
                search_depth="basic",
            )
            results = response.get("results", [])
            if not results:
                return "No relevant results found."

            per_doc_limit = MAX_INFO_CHARS // max(len(results), 1)
            parts = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                content = (r.get("content", ""))[:per_doc_limit]
                parts.append(f'Doc {i}(Title: "{title}") {content}')

            return " ".join(parts)

        except Exception as e:
            print(f"    ⚠ Tavily error: {e}")
            return "Search failed. No results available."


# ═══════════════════════════════════════════════════════════════════════════════
#  Null Backend (for baseline mode — never actually called, but keeps API clean)
# ═══════════════════════════════════════════════════════════════════════════════

class NullSearch:
    """Dummy backend that returns nothing. Used for the no-search baseline."""

    total_calls = 0

    def search(self, query: str) -> str:
        return "Search not available."