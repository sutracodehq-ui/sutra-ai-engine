"""
Knowledge Base Service — Manages vector storage and retrieval.

RAG-AI: Interfaces with Qdrant to store tenant-specific facts
and retrieve them during the chat pipeline. Embeddings via Ollama
(same model as the rest of the engine — see Settings.embedding_model).
"""

import asyncio
import logging
from typing import Any, List

from app.services.vector.qdrant_store import (
    embed_texts,
    get_qdrant_client,
    qdrant_collection_count,
    search_points,
    stable_point_id,
    upsert_points,
)

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Service to interface with vector database."""

    def _collection_name(self, tenant_id: int) -> str:
        return f"tenant_{tenant_id}_kb"

    async def add_documents(self, tenant_id: int, chunks: List[str], metadatas: List[dict], ids: List[str]):
        """Index document chunks into the vector store."""
        client = get_qdrant_client()
        if not client or not chunks:
            return

        def _run():
            coll = self._collection_name(tenant_id)
            vecs = embed_texts(chunks)
            if len(vecs) != len(chunks) or any(not v for v in vecs):
                logger.error("KnowledgeBase: embedding failed or empty vector")
                return
            pids = [stable_point_id(ids[i]) for i in range(len(ids))]
            payloads: List[dict[str, Any]] = []
            for i, ch in enumerate(chunks):
                pl = dict(metadatas[i]) if i < len(metadatas) else {}
                pl["document"] = ch
                payloads.append(pl)
            upsert_points(client, coll, pids, vecs, payloads)

        await asyncio.to_thread(_run)
        logger.info(f"Indexed {len(chunks)} chunks for tenant {tenant_id}")

    async def query(self, tenant_id: int, query_text: str, n_results: int = 3) -> List[str]:
        """Search for relevant chunks in the knowledge base."""
        try:
            client = get_qdrant_client()
            if not client:
                return []
            coll = self._collection_name(tenant_id)
            cnt = qdrant_collection_count(client, coll)
            if cnt == 0:
                return []
            qvecs = embed_texts([query_text])
            if not qvecs or not qvecs[0]:
                return []

            def _run():
                return search_points(client, coll, qvecs[0], limit=min(n_results, cnt))

            rows = await asyncio.to_thread(_run)
            out: List[str] = []
            for row in rows:
                pl = row.get("payload") or {}
                doc = pl.get("document", "")
                if doc:
                    out.append(doc)
            return out
        except Exception as e:
            logger.error(f"KnowledgeBase query failed: {e}")
            return []
