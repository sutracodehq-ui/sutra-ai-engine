"""
Brand Knowledge — Per-brand knowledge base for chatbot.

Software Factory Principle: Self-Learning + Data-Driven.

Each brand has its own knowledge base stored in ChromaDB.
When the AI doesn't know an answer and the brand owner provides one,
it gets stored here so the AI knows it next time.

Architecture:
    Brand Owner sets up FAQ/product info → ChromaDB collection per brand
    Customer asks question → Search brand's collection → Answer with context
    Unknown question → Owner answers → Stored in collection → AI learns
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class BrandKnowledge:
    """
    Per-brand knowledge base using ChromaDB.

    Each brand gets its own ChromaDB collection.
    Knowledge sources:
    1. Brand owner manually adds FAQ/info
    2. Auto-learned from owner escalation answers
    3. Imported from website/docs scraping
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        """Lazy-init ChromaDB client."""
        if self._client is None:
            import chromadb
            from urllib.parse import urlparse
            settings = get_settings()
            parsed = urlparse(settings.chromadb_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8000
            self._client = chromadb.HttpClient(host=host, port=port)
        return self._client

    def _get_collection(self, brand_id: str):
        """Get or create a brand-specific collection."""
        client = self._get_client()
        return client.get_or_create_collection(
            name=f"brand_{brand_id}_knowledge",
            metadata={"brand_id": brand_id, "type": "knowledge_base"},
        )

    # ─── Search ─────────────────────────────────────────────

    async def search(self, brand_id: str, query: str, n_results: int = 3) -> dict:
        """
        Search a brand's knowledge base for relevant answers.
        Returns context + confidence score.
        """
        try:
            collection = self._get_collection(brand_id)
            count = collection.count()

            if count == 0:
                return {"found": False, "confidence": 0.0, "context": ""}

            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, count),
            )

            if not results or not results["documents"] or not results["documents"][0]:
                return {"found": False, "confidence": 0.0, "context": ""}

            # Calculate confidence from distances
            distances = results["distances"][0] if results.get("distances") else [1.0]
            best_distance = min(distances)
            confidence = max(0, 1 - best_distance)

            # Build context from top results
            context_parts = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                source = meta.get("source", "knowledge_base")
                context_parts.append(
                    f"[{source}] {doc}"
                )

            context = "\n\n".join(context_parts)

            return {
                "found": True,
                "confidence": round(confidence, 3),
                "context": context,
                "results_count": len(results["documents"][0]),
            }

        except Exception as e:
            logger.warning(f"BrandKnowledge: search failed for {brand_id}: {e}")
            return {"found": False, "confidence": 0.0, "context": ""}

    # ─── Learn (Self-Teaching) ──────────────────────────────

    async def learn(self, brand_id: str, question: str, answer: str, source: str = "owner_answer") -> bool:
        """
        Store a new Q&A pair in the brand's knowledge base.
        Called when brand owner answers an escalated question.
        """
        try:
            collection = self._get_collection(brand_id)

            import hashlib
            doc_id = hashlib.md5(f"{question}:{answer}".encode()).hexdigest()[:16]

            # Store as a combined document for better retrieval
            document = f"Q: {question}\nA: {answer}"

            collection.upsert(
                ids=[f"kb_{doc_id}"],
                documents=[document],
                metadatas=[{
                    "question": question[:500],
                    "answer": answer[:2000],
                    "source": source,
                    "brand_id": brand_id,
                    "learned_at": datetime.now(timezone.utc).isoformat(),
                }],
            )

            logger.info(f"BrandKnowledge: learned new Q&A for brand {brand_id}")
            return True

        except Exception as e:
            logger.warning(f"BrandKnowledge: learn failed: {e}")
            return False

    # ─── Bulk Import ────────────────────────────────────────

    async def import_faq(self, brand_id: str, faq_items: list[dict]) -> dict:
        """
        Bulk import FAQ items into a brand's knowledge base.

        faq_items format: [{"question": "...", "answer": "..."}]
        """
        imported = 0
        for item in faq_items:
            success = await self.learn(
                brand_id=brand_id,
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                source="faq_import",
            )
            if success:
                imported += 1

        return {"imported": imported, "total": len(faq_items), "brand_id": brand_id}

    # ─── Stats ──────────────────────────────────────────────

    async def get_stats(self, brand_id: str) -> dict:
        """Get knowledge base stats for a brand."""
        try:
            collection = self._get_collection(brand_id)
            count = collection.count()
            return {
                "brand_id": brand_id,
                "total_knowledge_items": count,
            }
        except Exception:
            return {"brand_id": brand_id, "total_knowledge_items": 0}


# ─── Singleton ──────────────────────────────────────────────
_knowledge: BrandKnowledge | None = None


def get_brand_knowledge() -> BrandKnowledge:
    global _knowledge
    if _knowledge is None:
        _knowledge = BrandKnowledge()
    return _knowledge
