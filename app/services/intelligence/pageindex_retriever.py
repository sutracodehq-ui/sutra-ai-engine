"""
PageIndex Retriever — Vectorless, reasoning-based document retrieval.

Integrates the PageIndex framework for tree-structured document indexing
and LLM-powered reasoning-based retrieval. Designed for structured documents
like financial reports, legal contracts, lab reports, and academic content.

Architecture:
    Document (PDF/MD) → PageIndex → Hierarchical Tree Index (JSON)
    Query → LLM Tree Search → Relevant sections (with page references)

Unlike vector-based RAG (similarity ≠ relevance), PageIndex uses
LLM reasoning to navigate document structure — achieving human-like
retrieval with full traceability.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_pageindex_config() -> dict:
    """Load pageindex config from intelligence_config.yaml."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    rag_config = config.get("agentic_rag", {})
    return rag_config.get("pageindex", {})


class PageIndexRetriever:
    """
    Vectorless, reasoning-based document retrieval using PageIndex.

    Features:
    - Hierarchical tree indexing (semantic Table-of-Contents)
    - LLM reasoning-based tree search (no vector DB needed)
    - Traceable page/section references
    - Hybrid mode: combine with vector search for best of both

    Usage:
        retriever = get_pageindex_retriever()
        tree = await retriever.index_document("/path/to/report.pdf")
        results = await retriever.search(tree, "What is the revenue for Q3?")
    """

    def __init__(self, *, enabled: bool = True):
        self._enabled = enabled
        self._config = _load_pageindex_config()
        self._trees: dict[str, dict] = {}  # Cache: filepath → tree

    async def index_document(
        self,
        filepath: str,
        *,
        model: str | None = None,
        max_pages_per_node: int | None = None,
        max_tokens_per_node: int | None = None,
    ) -> dict:
        """
        Generate a PageIndex tree from a PDF or Markdown document.

        Returns the hierarchical tree structure as a dict.
        Caches the tree in memory for repeated queries.
        """
        if not self._enabled:
            return {}

        # Check cache first
        if filepath in self._trees:
            logger.debug(f"PageIndex: using cached tree for {filepath}")
            return self._trees[filepath]

        try:
            from pageindex import PageIndex

            pi = PageIndex(
                model=model or self._config.get("model", "gpt-4o-mini"),
                max_pages_per_node=max_pages_per_node or self._config.get("max_pages_per_node", 10),
                max_tokens_per_node=max_tokens_per_node or self._config.get("max_tokens_per_node", 20000),
            )

            file_path = Path(filepath)
            if file_path.suffix.lower() == ".pdf":
                tree = pi.index_pdf(str(file_path))
            elif file_path.suffix.lower() in (".md", ".markdown"):
                tree = pi.index_markdown(str(file_path))
            else:
                logger.warning(f"PageIndex: unsupported file type: {file_path.suffix}")
                return {}

            self._trees[filepath] = tree
            logger.info(
                f"PageIndex: indexed {filepath} "
                f"({self._count_nodes(tree)} nodes)"
            )
            return tree

        except ImportError:
            logger.warning(
                "PageIndex: 'pageindex' package not installed. "
                "Install with: pip install pageindex"
            )
            return {}
        except Exception as e:
            logger.error(f"PageIndex: indexing failed for {filepath}: {e}")
            return {}

    async def search(
        self,
        tree: dict,
        query: str,
        *,
        model: str | None = None,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Reasoning-based tree search over a PageIndex tree.

        Returns a list of relevant sections with page references:
        [
            {
                "title": "Section Title",
                "content": "Section content...",
                "start_page": 21,
                "end_page": 28,
                "relevance_reason": "Why this section is relevant"
            }
        ]
        """
        if not self._enabled or not tree:
            return []

        try:
            from pageindex import PageIndex

            pi = PageIndex(
                model=model or self._config.get("model", "gpt-4o-mini"),
            )

            results = pi.search(tree, query, top_k=n_results)

            structured = []
            for r in results:
                structured.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", r.get("summary", "")),
                    "start_page": r.get("start_index", 0),
                    "end_page": r.get("end_index", 0),
                    "node_id": r.get("node_id", ""),
                    "relevance_reason": r.get("reason", ""),
                })

            logger.info(
                f"PageIndex: search returned {len(structured)} results for: "
                f"{query[:60]}..."
            )
            return structured

        except ImportError:
            logger.warning("PageIndex: 'pageindex' package not installed")
            return []
        except Exception as e:
            logger.error(f"PageIndex: search failed: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        *,
        vector_results: list[dict] | None = None,
        tree: dict | None = None,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Merge PageIndex reasoning results with vector search results.

        Strategy:
        1. PageIndex results are prioritized (reasoning-based = higher precision)
        2. Vector results fill remaining slots (catches semantic matches)
        3. Deduplication by title/content similarity
        """
        all_results = []

        # PageIndex results first (higher precision)
        if tree:
            pi_results = await self.search(tree, query, n_results=n_results)
            for r in pi_results:
                r["source"] = "pageindex"
            all_results.extend(pi_results)

        # Vector results as supplement
        if vector_results:
            for r in vector_results:
                r["source"] = r.get("source", "vector")
            all_results.extend(vector_results)

        # Deduplicate and cap at n_results
        seen_titles = set()
        final = []
        for r in all_results:
            title = r.get("title", r.get("content", ""))[:100]
            if title not in seen_titles:
                seen_titles.add(title)
                final.append(r)
            if len(final) >= n_results:
                break

        return final

    def clear_cache(self, filepath: str | None = None) -> None:
        """Clear cached tree(s)."""
        if filepath:
            self._trees.pop(filepath, None)
        else:
            self._trees.clear()

    def _count_nodes(self, tree: dict) -> int:
        """Count total nodes in a tree structure."""
        count = 1
        for child in tree.get("nodes", []):
            count += self._count_nodes(child)
        return count


# ─── Singleton ──────────────────────────────────────────────

_retriever: PageIndexRetriever | None = None


def get_pageindex_retriever() -> PageIndexRetriever:
    """Get the global PageIndexRetriever singleton."""
    global _retriever
    if _retriever is None:
        config = _load_pageindex_config()
        _retriever = PageIndexRetriever(
            enabled=config.get("enabled", True),
        )
    return _retriever
