"""
Agent Memory Service — RAG-powered few-shot context injection.

Self-Learning Tier 1: Every successful agent response is stored as a vector
in Qdrant. On new requests, similar past responses are retrieved and
injected as few-shot examples, making the model progressively smarter.
"""

import hashlib
import logging

from app.services.vector.qdrant_store import (
    embed_texts,
    get_qdrant_client,
    qdrant_collection_count,
    search_points,
    stable_point_id,
    upsert_points,
)

logger = logging.getLogger(__name__)


class AgentMemoryService:
    """
    Qdrant-backed agent memory.

    Flow:
    1. On new request: query for similar past (prompt, response) pairs
    2. Inject top-K as few-shot examples into the message array
    3. After successful response: store the new (prompt, response) pair
    """

    COLLECTION_PREFIX = "agent_memory_"

    def __init__(self):
        self._enabled = True

    def _collection_name(self, agent_type: str) -> str:
        """Per-agent collection for isolated memory."""
        return f"{self.COLLECTION_PREFIX}{agent_type}"

    def _make_id(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    async def recall(self, agent_type: str, prompt: str, n_results: int = 2) -> list[dict]:
        """
        Retrieve similar past (prompt, response) pairs for few-shot injection.

        Uses Agentic RAG: decomposes complex prompts into multiple sub-queries
        for more targeted retrieval, then deduplicates and ranks results.

        Returns list of {"prompt": str, "response": str, "similarity": float} dicts.
        """
        if not self._enabled:
            return []

        try:
            client = get_qdrant_client()
            if not client:
                return []

            collection = self._collection_name(agent_type)
            if qdrant_collection_count(client, collection) == 0:
                return []

            queries = self._decompose_query(prompt)
            all_examples: dict[str, dict] = {}

            for query in queries:
                qvecs = embed_texts([query])
                if not qvecs or not qvecs[0]:
                    continue
                rows = search_points(
                    client,
                    collection,
                    qvecs[0],
                    limit=min(n_results, max(1, qdrant_collection_count(client, collection))),
                )

                for row in rows:
                    similarity = float(row.get("score", 0.0))
                    pl = row.get("payload") or {}
                    doc_id = str(row.get("id", ""))
                    doc_prompt = pl.get("document", pl.get("prompt", ""))
                    response = pl.get("response", "")

                    if similarity >= 0.7:
                        if doc_id not in all_examples or all_examples[doc_id]["similarity"] < similarity:
                            all_examples[doc_id] = {
                                "prompt": doc_prompt,
                                "response": response,
                                "similarity": round(similarity, 3),
                            }

            examples = sorted(all_examples.values(), key=lambda x: x["similarity"], reverse=True)[:n_results]

            if examples:
                logger.info(
                    f"AgentMemory: recalled {len(examples)} examples for {agent_type} "
                    f"(best similarity: {examples[0]['similarity']}, queries: {len(queries)})"
                )

            return examples

        except Exception as e:
            logger.warning(f"AgentMemory recall failed for {agent_type}: {e}")
            return []

    AFFINITY_GROUPS = {
        "education": [
            "education_guru", "edtech", "udise_compliance_advisor",
            "student_data_validator", "infrastructure_auditor",
            "udise_report_generator", "document_ocr_extractor",
            "school_selector",
        ],
        "marketing": [
            "copywriter", "seo", "social_media", "email_campaign",
            "ad_creative", "brand_auditor", "brand_advisor",
            "content_repurposer", "trend_spotter", "influencer_matcher",
        ],
        "finance": [
            "tax_advisor", "mutual_fund_advisor", "sip_calculator",
            "insurance_advisor", "pension_advisor",
        ],
        "health": [
            "symptom_checker", "medicine_info", "ayurveda_advisor",
            "elder_health_monitor", "pet_health",
        ],
    }

    def _get_affinity_peers(self, agent_type: str) -> list[str]:
        """Find peer agents in the same affinity group."""
        for group_agents in self.AFFINITY_GROUPS.values():
            if agent_type in group_agents:
                return [a for a in group_agents if a != agent_type]
        return []

    async def recall_cross_agent(
        self, agent_type: str, prompt: str, n_results: int = 2
    ) -> list[dict]:
        """
        Cross-agent memory recall: query related agents' memories
        when the primary agent has no relevant matches.

        Returns results tagged with source agent for transparency.
        """
        if not self._enabled:
            return []

        peers = self._get_affinity_peers(agent_type)
        if not peers:
            return []

        all_examples: list[dict] = []

        client = get_qdrant_client()
        if not client:
            return []

        for peer in peers:
            try:
                collection = self._collection_name(peer)
                if qdrant_collection_count(client, collection) == 0:
                    continue

                qvecs = embed_texts([prompt])
                if not qvecs or not qvecs[0]:
                    continue
                rows = search_points(
                    client,
                    collection,
                    qvecs[0],
                    limit=min(1, qdrant_collection_count(client, collection)),
                )

                for row in rows:
                    similarity = float(row.get("score", 0.0))
                    pl = row.get("payload") or {}
                    if similarity >= 0.75:
                        all_examples.append({
                            "prompt": pl.get("document", pl.get("prompt", "")),
                            "response": pl.get("response", ""),
                            "similarity": round(similarity, 3),
                            "source_agent": peer,
                        })
            except Exception as e:
                logger.debug(f"Cross-agent recall from {peer} skipped: {e}")

        all_examples.sort(key=lambda x: x["similarity"], reverse=True)
        result = all_examples[:n_results]

        if result:
            sources = [r["source_agent"] for r in result]
            logger.info(
                f"AgentMemory: cross-agent recall for {agent_type} "
                f"→ {len(result)} examples from {sources}"
            )

        return result

    def _decompose_query(self, prompt: str) -> list[str]:
        """
        Decompose a complex prompt into multiple sub-queries for better retrieval.

        Agentic RAG: Instead of searching for the entire prompt as one vector,
        break it into semantic chunks that might match different past examples.
        """
        queries = [prompt]

        sentences = [s.strip() for s in prompt.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        if len(sentences) > 1:
            for sentence in sentences:
                if len(sentence.split()) >= 5:
                    queries.append(sentence)

        import re
        quoted = re.findall(r'"([^"]+)"', prompt)
        queries.extend(quoted)

        topic_matches = re.findall(r'(?:about|for|on|regarding)\s+(.+?)(?:\.|,|$)', prompt, re.I)
        queries.extend(topic_matches)

        seen = set()
        unique_queries = []
        for q in queries:
            q_clean = q.strip().lower()
            if q_clean and q_clean not in seen:
                seen.add(q_clean)
                unique_queries.append(q)
            if len(unique_queries) >= 4:
                break

        return unique_queries

    async def remember(self, agent_type: str, prompt: str, response: str, quality_score: float = 1.0) -> None:
        """
        Store a successful (prompt, response) pair for future recall.

        Only stores responses with quality_score >= 0.7 (accepted or lightly edited).
        """
        if not self._enabled or quality_score < 0.7:
            return

        try:
            client = get_qdrant_client()
            if not client:
                return

            collection = self._collection_name(agent_type)
            doc_id = self._make_id(prompt)
            truncated_response = response[:30000] if len(response) > 30000 else response

            vecs = embed_texts([prompt])
            if not vecs or not vecs[0]:
                return

            pid = stable_point_id(doc_id)
            payload = {
                "document": prompt,
                "prompt": prompt,
                "response": truncated_response,
                "quality_score": str(quality_score),
                "agent_type": agent_type,
            }
            upsert_points(client, collection, [pid], [vecs[0]], [payload])

            logger.debug(f"AgentMemory: stored example for {agent_type}, id={doc_id[:16]}...")

        except Exception as e:
            logger.warning(f"AgentMemory store failed for {agent_type}: {e}")

    async def get_stats(self, agent_type: str) -> dict:
        """Get memory stats for an agent."""
        try:
            client = get_qdrant_client()
            if not client:
                return {"agent_type": agent_type, "total_memories": 0}
            collection = self._collection_name(agent_type)
            return {
                "agent_type": agent_type,
                "total_memories": qdrant_collection_count(client, collection),
            }
        except Exception as e:
            return {"agent_type": agent_type, "total_memories": 0, "error": str(e)}


_memory: AgentMemoryService | None = None


def get_agent_memory() -> AgentMemoryService:
    """Get the global AgentMemoryService instance."""
    global _memory
    if _memory is None:
        _memory = AgentMemoryService()
    return _memory
