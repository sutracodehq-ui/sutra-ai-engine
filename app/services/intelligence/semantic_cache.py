"""
Semantic Cache — ChromaDB vector-similarity cache.

Performance impact: finds similar-enough past prompts and returns cached responses.
Unlike PromptCache (exact match), this catches paraphrased/reformulated prompts.
Typically hits 5-15% of traffic that PromptCache misses.
"""

import hashlib
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    ChromaDB-based semantic similarity cache.

    1. Embed the user prompt
    2. Query ChromaDB for similar prompts (cosine similarity > threshold)
    3. If hit → return cached response
    4. If miss → after LLM call, store prompt + response for future hits
    """

    COLLECTION_NAME = "sutra_prompt_cache"

    def __init__(self, chromadb_client, *, enabled: bool = True, similarity_threshold: float = 0.92):
        self._client = chromadb_client
        self._enabled = enabled
        self._threshold = similarity_threshold
        self._collection = None

    async def _get_collection(self):
        """Lazy-init the ChromaDB collection."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _make_id(self, agent_type: str, prompt: str) -> str:
        """Generate a unique ID for a prompt."""
        payload = f"{agent_type}:{prompt}"
        return hashlib.md5(payload.encode()).hexdigest()

    async def get(self, agent_type: str, prompt: str) -> Optional[dict]:
        """Query for a semantically similar cached prompt."""
        if not self._enabled:
            return None

        try:
            collection = await self._get_collection()
            results = collection.query(
                query_texts=[prompt],
                n_results=1,
                where={"agent_type": agent_type},
            )

            if (
                results
                and results["distances"]
                and results["distances"][0]
                and results["distances"][0][0] <= (1 - self._threshold)  # cosine distance
            ):
                metadata = results["metadatas"][0][0]
                cached_response = json.loads(metadata.get("response", "{}"))
                logger.info(
                    f"SemanticCache HIT: similarity={1 - results['distances'][0][0]:.3f}, "
                    f"agent={agent_type}"
                )
                return cached_response

        except Exception as e:
            logger.warning(f"SemanticCache query error: {e}")

        return None

    async def set(self, agent_type: str, prompt: str, response: dict) -> None:
        """Store a prompt + response for future similarity matching."""
        if not self._enabled:
            return

        try:
            collection = await self._get_collection()
            doc_id = self._make_id(agent_type, prompt)

            collection.upsert(
                ids=[doc_id],
                documents=[prompt],
                metadatas=[{
                    "agent_type": agent_type,
                    "response": json.dumps(response),
                }],
            )
            logger.debug(f"SemanticCache SET: agent={agent_type}, id={doc_id[:16]}...")

        except Exception as e:
            logger.warning(f"SemanticCache store error: {e}")
