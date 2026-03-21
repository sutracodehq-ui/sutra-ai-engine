"""
Agent Memory Service — RAG-powered few-shot context injection.

Self-Learning Tier 1: Every successful agent response is stored as a vector
in ChromaDB. On new requests, similar past responses are retrieved and
injected as few-shot examples, making the model progressively smarter.
"""

import hashlib
import json
import logging
from typing import Optional

import chromadb

from app.config import get_settings

logger = logging.getLogger(__name__)


class AgentMemoryService:
    """
    ChromaDB-backed agent memory.

    Flow:
    1. On new request: query for similar past (prompt, response) pairs
    2. Inject top-K as few-shot examples into the message array
    3. After successful response: store the new (prompt, response) pair
    """

    COLLECTION_PREFIX = "agent_memory_"

    def __init__(self):
        settings = get_settings()
        self._client = chromadb.HttpClient(host=settings.chromadb_url.replace("http://", "").split(":")[0],
                                           port=int(settings.chromadb_url.split(":")[-1]))
        self._enabled = True

    def _collection_name(self, agent_type: str) -> str:
        """Per-agent collection for isolated memory."""
        return f"{self.COLLECTION_PREFIX}{agent_type}"

    def _make_id(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    async def recall(self, agent_type: str, prompt: str, n_results: int = 2) -> list[dict]:
        """
        Retrieve similar past (prompt, response) pairs for few-shot injection.

        Returns list of {"prompt": str, "response": str} dicts.
        """
        if not self._enabled:
            return []

        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name(agent_type),
                metadata={"hnsw:space": "cosine"},
            )

            if collection.count() == 0:
                return []

            results = collection.query(
                query_texts=[prompt],
                n_results=min(n_results, collection.count()),
            )

            examples = []
            if results and results["metadatas"] and results["metadatas"][0]:
                for i, meta in enumerate(results["metadatas"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    similarity = 1 - distance

                    # Only use examples with >70% similarity
                    if similarity >= 0.7:
                        examples.append({
                            "prompt": results["documents"][0][i],
                            "response": meta.get("response", ""),
                            "similarity": round(similarity, 3),
                        })

            if examples:
                logger.info(
                    f"AgentMemory: recalled {len(examples)} examples for {agent_type} "
                    f"(best similarity: {examples[0]['similarity']})"
                )

            return examples

        except Exception as e:
            logger.warning(f"AgentMemory recall failed for {agent_type}: {e}")
            return []

    async def remember(self, agent_type: str, prompt: str, response: str, quality_score: float = 1.0) -> None:
        """
        Store a successful (prompt, response) pair for future recall.

        Only stores responses with quality_score >= 0.7 (accepted or lightly edited).
        """
        if not self._enabled or quality_score < 0.7:
            return

        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name(agent_type),
                metadata={"hnsw:space": "cosine"},
            )

            doc_id = self._make_id(prompt)

            # Truncate response to fit ChromaDB metadata limits (~32KB)
            truncated_response = response[:30000] if len(response) > 30000 else response

            collection.upsert(
                ids=[doc_id],
                documents=[prompt],
                metadatas=[{
                    "response": truncated_response,
                    "quality_score": str(quality_score),
                    "agent_type": agent_type,
                }],
            )

            logger.debug(f"AgentMemory: stored example for {agent_type}, id={doc_id[:16]}...")

        except Exception as e:
            logger.warning(f"AgentMemory store failed for {agent_type}: {e}")

    async def get_stats(self, agent_type: str) -> dict:
        """Get memory stats for an agent."""
        try:
            collection = self._client.get_or_create_collection(
                name=self._collection_name(agent_type),
            )
            return {
                "agent_type": agent_type,
                "total_memories": collection.count(),
            }
        except Exception as e:
            return {"agent_type": agent_type, "total_memories": 0, "error": str(e)}


# ─── Singleton ──────────────────────────────────────────────────

_memory: AgentMemoryService | None = None


def get_agent_memory() -> AgentMemoryService:
    """Get the global AgentMemoryService instance."""
    global _memory
    if _memory is None:
        _memory = AgentMemoryService()
    return _memory
