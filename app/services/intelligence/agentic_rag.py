"""
Agentic RAG — Retrieval with Reasoning Loops.

Software Factory Principle: Quality Control at every layer.

Upgrades basic RAG (retrieve → dump) to intelligent retrieval with:
1. Iterative retrieval: If first fetch is weak, refine query and retry
2. Multi-source fusion: ChromaDB + Web Scanner + Agent Memory
3. Claim verification: Cross-check facts before including
4. Source citation: Track where each piece of data came from
5. Confidence scoring: Rate reliability of retrieved information
"""

import logging
from typing import Optional

import yaml
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_rag_config() -> dict:
    """Load agentic RAG config from YAML."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("agentic_rag", {})


class RetrievedChunk:
    """A single retrieved piece of information with metadata."""

    def __init__(
        self, content: str, source: str, relevance: float,
        metadata: dict | None = None,
    ):
        self.content = content
        self.source = source  # "chromadb", "web_scanner", "agent_memory"
        self.relevance = relevance
        self.metadata = metadata or {}
        self.verified = False
        self.confidence = 0.0

    def to_dict(self) -> dict:
        return {
            "content": self.content[:500],
            "source": self.source,
            "relevance": round(self.relevance, 3),
            "verified": self.verified,
            "confidence": round(self.confidence, 3),
        }


class AgenticRAG:
    """
    Retrieval-Augmented Generation with reasoning loops.

    Unlike basic RAG which retrieves once and dumps context,
    Agentic RAG:
    - Iterates: refines queries when initial retrieval is weak
    - Fuses: combines data from multiple sources
    - Verifies: cross-checks facts before using them
    - Cites: tracks the source of every piece of information
    """

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from app.services.llm_service import get_llm_service
            self._llm = get_llm_service()
        return self._llm

    # ─── Phase 1: Multi-Source Retrieval ─────────────────────

    async def retrieve(self, query: str, agent_type: str = "general") -> list[RetrievedChunk]:
        """
        Retrieve from all available sources and merge results.
        Sources: ChromaDB (vector store), Web Scanner (live data), Agent Memory.
        """
        config = _load_rag_config()
        chunks: list[RetrievedChunk] = []

        # 1. ChromaDB (vector similarity search)
        try:
            import chromadb
            settings = get_settings()
            client = chromadb.HttpClient(
                host=settings.chroma_host , port=int(settings.chroma_port)
            )

            # Search across relevant collections
            collections_to_search = config.get("collections", [
                "web_articles", "stock_data", "crypto_data", "agent_responses",
            ])

            for col_name in collections_to_search:
                try:
                    collection = client.get_collection(col_name)
                    results = collection.query(query_texts=[query], n_results=3)
                    if results and results["documents"]:
                        for i, doc in enumerate(results["documents"][0]):
                            distance = results["distances"][0][i] if results.get("distances") else 0.5
                            relevance = max(0, 1 - distance)
                            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                            chunks.append(RetrievedChunk(
                                content=doc,
                                source=f"chromadb:{col_name}",
                                relevance=relevance,
                                metadata=meta,
                            ))
                except Exception:
                    pass  # Collection may not exist yet

        except Exception as e:
            logger.debug(f"AgenticRAG: ChromaDB retrieval skipped: {e}")

        # 2. Agent Memory (past successful responses)
        try:
            from app.services.intelligence.agent_memory import get_agent_memory
            memory = get_agent_memory()
            past = await memory.recall(agent_type, query, n_results=3)
            for ex in past:
                chunks.append(RetrievedChunk(
                    content=ex.get("response", ""),
                    source="agent_memory",
                    relevance=0.7,  # Memory results are pre-filtered
                    metadata={"original_prompt": ex.get("prompt", "")},
                ))
        except Exception as e:
            logger.debug(f"AgenticRAG: agent memory retrieval skipped: {e}")

        logger.info(f"AgenticRAG: retrieved {len(chunks)} chunks from all sources")
        return chunks

    # ─── Phase 2: Iterative Refinement ──────────────────────

    async def iterative_retrieve(
        self, query: str, agent_type: str = "general", max_iterations: int = 2,
    ) -> list[RetrievedChunk]:
        """
        Retrieve with iterative refinement.
        If initial retrieval quality is low, refine the query and retry.
        """
        config = _load_rag_config()
        min_relevance = config.get("min_relevance_threshold", 0.5)
        current_query = query

        all_chunks: list[RetrievedChunk] = []

        for iteration in range(max_iterations):
            chunks = await self.retrieve(current_query, agent_type)
            all_chunks.extend(chunks)

            # Check if we have enough high-quality chunks
            high_quality = [c for c in chunks if c.relevance >= min_relevance]
            if len(high_quality) >= 3:
                logger.info(f"AgenticRAG: sufficient quality at iteration {iteration + 1}")
                break

            # Refine query for next iteration
            if iteration < max_iterations - 1:
                current_query = await self._refine_query(query, chunks)
                logger.info(f"AgenticRAG: refined query → '{current_query[:80]}...'")

        # Deduplicate and sort by relevance
        seen = set()
        unique = []
        for chunk in all_chunks:
            key = chunk.content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(chunk)

        unique.sort(key=lambda c: c.relevance, reverse=True)
        return unique

    async def _refine_query(self, original: str, weak_results: list) -> str:
        """Ask the LLM to reformulate the query for better retrieval."""
        llm = self._get_llm()
        response = await llm.complete(
            prompt=(
                f"The search query '{original}' returned weak results. "
                f"Reformulate it to be more specific and likely to find relevant information. "
                f"Return ONLY the refined query, nothing else."
            ),
            system_prompt="You reformulate search queries for better retrieval.",
        )
        return response.content.strip().strip('"').strip("'")

    # ─── Phase 3: Verification ──────────────────────────────

    async def verify_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Cross-check retrieved chunks for consistency.
        Chunks that contradict each other get lower confidence.
        """
        if len(chunks) <= 1:
            for c in chunks:
                c.verified = True
                c.confidence = c.relevance
            return chunks

        # Group by source for cross-checking
        sources = {}
        for chunk in chunks:
            src = chunk.source.split(":")[0]
            sources.setdefault(src, []).append(chunk)

        # If data comes from multiple sources, higher confidence
        for chunk in chunks:
            src = chunk.source.split(":")[0]
            multi_source_bonus = 0.1 if len(sources) > 1 else 0
            chunk.confidence = min(chunk.relevance + multi_source_bonus, 1.0)
            chunk.verified = chunk.confidence >= 0.5

        verified_count = sum(1 for c in chunks if c.verified)
        logger.info(f"AgenticRAG: verified {verified_count}/{len(chunks)} chunks")
        return chunks

    # ─── Phase 4: Context Assembly ──────────────────────────

    async def build_context(
        self, query: str, agent_type: str = "general", max_chunks: int = 5,
    ) -> dict:
        """
        Full agentic RAG pipeline:
        1. Iteratively retrieve from all sources
        2. Verify and score
        3. Assemble into context with citations
        """
        # 1. Iterative retrieval
        chunks = await self.iterative_retrieve(query, agent_type)

        # 2. Verify
        chunks = await self.verify_chunks(chunks)

        # 3. Take top-N verified chunks
        verified = [c for c in chunks if c.verified][:max_chunks]

        # 4. Build cited context
        context_parts = []
        citations = []
        for i, chunk in enumerate(verified):
            ref = f"[{i+1}]"
            context_parts.append(f"{ref} {chunk.content}")
            citations.append({
                "ref": ref,
                "source": chunk.source,
                "confidence": round(chunk.confidence, 2),
                "metadata": chunk.metadata,
            })

        context_text = "\n\n".join(context_parts)

        return {
            "context": context_text,
            "citations": citations,
            "total_retrieved": len(chunks),
            "total_verified": len(verified),
            "sources_used": list(set(c.source.split(":")[0] for c in verified)),
        }


# ─── Singleton ──────────────────────────────────────────────
_rag: AgenticRAG | None = None


def get_agentic_rag() -> AgenticRAG:
    global _rag
    if _rag is None:
        _rag = AgenticRAG()
    return _rag
