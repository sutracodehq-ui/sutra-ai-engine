"""
Agent Learning System — Continuous improvement through feedback and memory.

How agents keep learning:
1. Feedback Loop: Users rate responses (👍/👎) + provide corrections
2. Agent Memory: Good responses stored in per-agent vector DB (Qdrant)
3. Quality Score: Track accuracy over time, auto-flag degrading agents
4. Context Enrichment: Inject top past learnings into prompts
5. Correction Learning: Bad responses + corrections = new training examples

Flow:
    User gives feedback → Store in Qdrant → 
    Next call: inject relevant past learnings into system prompt →
    Agent gets smarter over time
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """A single feedback record."""
    feedback_id: str
    agent_id: str
    tenant_id: str
    prompt: str
    response: str
    rating: int            # 1 (terrible) to 5 (excellent)
    correction: str = ""   # User's correction if bad
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentQuality:
    """Quality metrics for an agent."""
    agent_id: str
    total_ratings: int = 0
    avg_rating: float = 0.0
    positive_count: int = 0    # rating >= 4
    negative_count: int = 0    # rating <= 2
    correction_count: int = 0
    trend: str = "stable"      # improving, stable, degrading


class AgentLearningSystem:
    """
    Continuous learning engine.
    
    Stores feedback and good examples in Qdrant per agent.
    Before each agent call, injects relevant past learnings
    to improve response quality.
    """

    def __init__(self):
        self._feedback: dict[str, list[FeedbackEntry]] = {}  # agent_id → [feedbacks]
        self._quality: dict[str, AgentQuality] = {}
        self._learnings: dict[str, list[dict]] = {}  # agent_id → [{prompt, good_response}]
        self._corrections: dict[str, list[dict]] = {}  # agent_id → [{prompt, bad_response, correction}]
        self._vector_client = None

    def _init_vector_store(self):
        """Initialize Qdrant client for storing learnings."""
        if self._vector_client is not None:
            return
        try:
            from app.services.vector.qdrant_store import get_qdrant_client

            self._vector_client = get_qdrant_client()
            if self._vector_client:
                logger.info("AgentLearning: Qdrant initialized for learning memory")
            else:
                logger.warning("AgentLearning: Qdrant URL not configured, vector learnings disabled")
        except Exception as e:
            logger.warning("AgentLearning: Qdrant unavailable: %s", e)

    def submit_feedback(
        self,
        agent_id: str,
        tenant_id: str,
        prompt: str,
        response: str,
        rating: int,
        correction: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Submit user feedback on an agent response.
        
        Rating scale: 1-5 (1=terrible, 5=excellent)
        Correction: What the response SHOULD have been
        """
        import uuid

        feedback_id = str(uuid.uuid4())[:12]
        entry = FeedbackEntry(
            feedback_id=feedback_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            prompt=prompt,
            response=response,
            rating=min(max(rating, 1), 5),
            correction=correction,
            tags=tags or [],
        )

        self._feedback.setdefault(agent_id, []).append(entry)

        # ── Learn from this feedback ──
        if rating >= 4:
            # Good response → store as example
            self._learnings.setdefault(agent_id, []).append({
                "prompt": prompt,
                "good_response": response,
                "rating": rating,
            })
            self._store_in_vector_db(agent_id, prompt, response, "good_example")
        elif rating <= 2 and correction:
            # Bad response with correction → learn the fix
            self._corrections.setdefault(agent_id, []).append({
                "prompt": prompt,
                "bad_response": response,
                "correction": correction,
            })
            self._store_in_vector_db(agent_id, prompt, correction, "correction")

        # Update quality metrics
        self._update_quality(agent_id)

        logger.info(
            f"AgentLearning: feedback for '{agent_id}' — "
            f"rating={rating}, has_correction={bool(correction)}"
        )
        return feedback_id

    def get_learnings_for_prompt(
        self,
        agent_id: str,
        prompt: str,
        max_examples: int = 3,
    ) -> str:
        """
        Get relevant past learnings to inject into the agent's context.
        
        Returns a formatted string to prepend to the system prompt.
        """
        learnings = []

        # ── Try vector search first (most accurate) ──
        vector_results = self._search_vector_db(agent_id, prompt, max_examples)
        if vector_results:
            for doc in vector_results:
                learnings.append(doc)

        # ── Fallback: keyword matching from in-memory ──
        else:
            prompt_lower = prompt.lower()
            examples = self._learnings.get(agent_id, [])
            corrections = self._corrections.get(agent_id, [])

            for ex in examples[-50:]:  # Check recent 50
                if any(word in ex["prompt"].lower() for word in prompt_lower.split()[:5]):
                    learnings.append(f"Good example for similar query:\nQ: {ex['prompt'][:200]}\nA: {ex['good_response'][:300]}")

            for cor in corrections[-50:]:
                if any(word in cor["prompt"].lower() for word in prompt_lower.split()[:5]):
                    learnings.append(f"Correction from past mistake:\nQ: {cor['prompt'][:200]}\n❌ Bad: {cor['bad_response'][:150]}\n✅ Correct: {cor['correction'][:300]}")

        # ── Also include cross-teachings ──
        cross_teachings = [
            ex for ex in self._learnings.get(agent_id, [])
            if ex.get("prompt", "").startswith("[teaching:")
        ]
        for ct in cross_teachings[-5:]:
            learnings.append(ct.get("good_response", ""))

        if not learnings:
            return ""

        header = "\n--- LEARNINGS FROM PAST FEEDBACK & CROSS-TEACHINGS (improve your response using these) ---\n"
        return header + "\n\n".join(learnings[:max_examples]) + "\n--- END LEARNINGS ---\n"

    def get_quality(self, agent_id: str) -> dict:
        """Get quality metrics for an agent."""
        q = self._quality.get(agent_id)
        if not q:
            return {"agent_id": agent_id, "status": "no_data"}

        return {
            "agent_id": q.agent_id,
            "total_ratings": q.total_ratings,
            "avg_rating": round(q.avg_rating, 2),
            "positive_rate": f"{(q.positive_count / max(q.total_ratings, 1)) * 100:.0f}%",
            "corrections": q.correction_count,
            "trend": q.trend,
            "status": "healthy" if q.avg_rating >= 3.5 else "needs_attention",
        }

    def get_all_quality(self) -> list[dict]:
        """Get quality for all agents (admin dashboard)."""
        return [self.get_quality(aid) for aid in self._quality]

    def get_degrading_agents(self) -> list[dict]:
        """Get agents with declining quality (admin alert)."""
        return [
            self.get_quality(aid)
            for aid, q in self._quality.items()
            if q.trend == "degrading" or q.avg_rating < 3.0
        ]

    def _update_quality(self, agent_id: str):
        """Recalculate quality metrics for an agent."""
        feedbacks = self._feedback.get(agent_id, [])
        if not feedbacks:
            return

        ratings = [f.rating for f in feedbacks]
        q = self._quality.get(agent_id, AgentQuality(agent_id=agent_id))

        q.total_ratings = len(ratings)
        q.avg_rating = sum(ratings) / len(ratings)
        q.positive_count = sum(1 for r in ratings if r >= 4)
        q.negative_count = sum(1 for r in ratings if r <= 2)
        q.correction_count = sum(1 for f in feedbacks if f.correction)

        # Trend: compare last 10 vs previous 10
        if len(ratings) >= 20:
            recent = sum(ratings[-10:]) / 10
            older = sum(ratings[-20:-10]) / 10
            if recent > older + 0.3:
                q.trend = "improving"
            elif recent < older - 0.3:
                q.trend = "degrading"
            else:
                q.trend = "stable"

        self._quality[agent_id] = q

    def _store_in_vector_db(self, agent_id: str, prompt: str, content: str, doc_type: str):
        """Store learning in Qdrant for semantic search."""
        self._init_vector_store()
        if not self._vector_client:
            return

        try:
            from app.services.vector.qdrant_store import embed_texts, stable_point_id, upsert_points

            collection_name = f"agent_learnings_{agent_id}"
            doc_text = f"Q: {prompt}\nA: {content}"
            doc_id = hashlib.md5(f"{prompt}{content}".encode()).hexdigest()
            vecs = embed_texts([doc_text])
            if not vecs or not vecs[0]:
                return
            pid = stable_point_id(doc_id)
            payload = {
                "document": doc_text,
                "type": doc_type,
                "agent_id": agent_id,
                "timestamp": time.time(),
            }
            upsert_points(self._vector_client, collection_name, [pid], [vecs[0]], [payload])
        except Exception as e:
            logger.debug(f"AgentLearning: vector store write failed: {e}")

    def _search_vector_db(self, agent_id: str, prompt: str, max_results: int) -> list[str]:
        """Search Qdrant for relevant past learnings."""
        self._init_vector_store()
        if not self._vector_client:
            return []

        try:
            from app.services.vector.qdrant_store import (
                embed_texts,
                qdrant_collection_count,
                search_points,
            )

            collection_name = f"agent_learnings_{agent_id}"
            cnt = qdrant_collection_count(self._vector_client, collection_name)
            if cnt == 0:
                return []
            qvecs = embed_texts([prompt])
            if not qvecs or not qvecs[0]:
                return []
            rows = search_points(
                self._vector_client,
                collection_name,
                qvecs[0],
                limit=min(max_results, cnt),
            )
            return [(r.get("payload") or {}).get("document", "") for r in rows if (r.get("payload") or {}).get("document")]
        except Exception as e:
            logger.debug(f"AgentLearning: vector search failed: {e}")
            return []


    def get_teaching_effectiveness(self, agent_id: str) -> dict:
        """Measure quality improvement after receiving cross-teachings."""
        quality = self.get_quality(agent_id)
        cross_teachings = [
            ex for ex in self._learnings.get(agent_id, [])
            if ex.get("prompt", "").startswith("[teaching:")
        ]

        return {
            "agent_id": agent_id,
            "teachings_received": len(cross_teachings),
            "current_avg_rating": quality.get("avg_rating", 0),
            "quality_trend": quality.get("trend", "unknown"),
            "status": quality.get("status", "no_data"),
        }

    def get_alliance_learnings_summary(self) -> list[dict]:
        """Get teaching effectiveness for all agents that received teachings."""
        agents_with_teachings = [
            agent_id for agent_id, examples in self._learnings.items()
            if any(ex.get("prompt", "").startswith("[teaching:") for ex in examples)
        ]
        return [self.get_teaching_effectiveness(aid) for aid in agents_with_teachings]


# ─── Singleton ──────────────────────────────────────────────

_learning: AgentLearningSystem | None = None

def get_agent_learning() -> AgentLearningSystem:
    global _learning
    if _learning is None:
        _learning = AgentLearningSystem()
    return _learning
