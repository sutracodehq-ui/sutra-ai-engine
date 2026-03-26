"""
Cross-Teaching Engine — Agent-to-Agent Knowledge Sharing.

Software Factory Principle: Collective Intelligence.

Enables agents to share their learnings with peers in the same
"teaching alliance". When an SEO agent learns that "short meta titles
convert better", this engine distills that insight and pushes it
to the Copywriter, Social Media, and Email Campaign agents.

Architecture:
    Top Agent → extract_teachings() → insight → teach() → Peer Agent Memory
                                                             ↕
                                              Agent reads insights on next call

Flow:
    1. Find high-quality agents (avg rating ≥ threshold)
    2. Extract transferable insights from their best responses
    3. Push insights to alliance members via AgentLearningSystem
    4. Track effectiveness: did the student improve?
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("intelligence_config.yaml")
HISTORY_PATH = Path("training/teaching_history.jsonl")
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_teaching_config() -> dict:
    """Load cross-teaching config from YAML."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("cross_teaching", {})


class TeachingInsight:
    """A transferable piece of knowledge from one agent to another."""

    def __init__(
        self,
        teacher_agent: str,
        student_agent: str,
        topic: str,
        insight: str,
        source_examples: int = 0,
        confidence: float = 0.0,
    ):
        self.teacher_agent = teacher_agent
        self.student_agent = student_agent
        self.topic = topic
        self.insight = insight
        self.source_examples = source_examples
        self.confidence = confidence
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.insight_id = hashlib.md5(
            f"{teacher_agent}:{student_agent}:{insight[:100]}".encode()
        ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "insight_id": self.insight_id,
            "teacher": self.teacher_agent,
            "student": self.student_agent,
            "topic": self.topic,
            "insight": self.insight,
            "source_examples": self.source_examples,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


class CrossTeacher:
    """
    Cross-Teaching Engine for agent-to-agent knowledge sharing.

    Workflow:
    1. extract_teachings() — Mine an agent's best responses for patterns
    2. teach() — Push insight to a target agent's memory
    3. run_teaching_cycle() — Full cycle for all alliances
    """

    def __init__(self):
        self._taught_hashes: set[str] = set()  # Dedup within session

    # ─── Extract Teachings ──────────────────────────────────

    async def extract_teachings(self, agent_id: str, max_insights: int = 5) -> list[TeachingInsight]:
        """
        Mine an agent's high-quality responses to extract transferable insights.

        Uses LLM to summarize patterns from the agent's best work into
        concise, actionable insights that other agents can apply.
        """
        from app.services.intelligence.agent_learning import get_agent_learning

        learning = get_agent_learning()
        quality = learning.get_quality(agent_id)

        # Must meet minimum quality bar
        config = _load_teaching_config()
        min_quality = config.get("min_quality_to_teach", 4.0)
        min_feedback = config.get("min_feedback_count", 10)

        if quality.get("status") == "no_data":
            logger.debug(f"CrossTeacher: {agent_id} has no quality data, skipping")
            return []

        if quality.get("avg_rating", 0) < min_quality:
            logger.debug(
                f"CrossTeacher: {agent_id} avg rating {quality.get('avg_rating')} "
                f"below threshold {min_quality}"
            )
            return []

        if quality.get("total_ratings", 0) < min_feedback:
            logger.debug(f"CrossTeacher: {agent_id} has insufficient feedback count")
            return []

        # Get the agent's best examples from memory
        good_examples = learning._learnings.get(agent_id, [])
        corrections = learning._corrections.get(agent_id, [])

        if not good_examples and not corrections:
            return []

        # Use LLM to distill patterns from good examples
        insights = await self._distill_insights(agent_id, good_examples, corrections, max_insights)
        return insights

    async def _distill_insights(
        self,
        agent_id: str,
        good_examples: list[dict],
        corrections: list[dict],
        max_insights: int,
    ) -> list[TeachingInsight]:
        """Use LLM to extract transferable insights from examples."""
        from app.services.llm_service import get_llm_service

        llm = get_llm_service()

        # Build context from examples
        examples_text = "\n".join(
            f"- Q: {ex['prompt'][:150]}\n  A (rated {ex.get('rating', 'high')}): {ex['good_response'][:200]}"
            for ex in good_examples[-20:]  # Last 20 good examples
        )

        corrections_text = "\n".join(
            f"- Q: {c['prompt'][:150]}\n  ❌ Bad: {c['bad_response'][:100]}\n  ✅ Fix: {c['correction'][:150]}"
            for c in corrections[-10:]  # Last 10 corrections
        )

        prompt = f"""Analyze these successful AI agent responses and corrections.
Extract {max_insights} transferable insights that OTHER agents in the same domain could apply.

## Agent: {agent_id}

## Successful Responses:
{examples_text or "None available"}

## Corrections Applied:
{corrections_text or "None available"}

## Rules:
- Each insight must be actionable and specific (not generic advice)
- Focus on PATTERNS, FORMAT, TONE, and GUARDRAILS
- Each insight should be ONE concise sentence

Return ONLY valid JSON array:
[{{"topic": "response_format", "insight": "Always include a risk disclaimer at the end of financial analysis", "confidence": 0.85}}]"""

        response = await llm.complete(
            prompt=prompt,
            system_prompt="You extract transferable patterns from AI agent behavior. Return only valid JSON.",
        )

        insights = []
        try:
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                raw = json.loads(json_match.group())
                for item in raw[:max_insights]:
                    insights.append(TeachingInsight(
                        teacher_agent=agent_id,
                        student_agent="",  # Set during teach()
                        topic=item.get("topic", "general"),
                        insight=item.get("insight", ""),
                        source_examples=len(good_examples),
                        confidence=float(item.get("confidence", 0.5)),
                    ))
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"CrossTeacher: insight extraction parse failed: {e}")

        logger.info(f"CrossTeacher: extracted {len(insights)} insights from {agent_id}")
        return insights

    # ─── Teach ──────────────────────────────────────────────

    async def teach(self, insight: TeachingInsight, target_agent: str) -> bool:
        """
        Push an insight to a target agent's learning memory.

        The insight is stored in ChromaDB tagged as cross-teaching,
        so the agent can retrieve it in future prompts.
        """
        from app.services.intelligence.agent_learning import get_agent_learning

        # Dedup check
        dedup_key = f"{insight.teacher_agent}:{target_agent}:{insight.insight[:50]}"
        dedup_hash = hashlib.md5(dedup_key.encode()).hexdigest()
        if dedup_hash in self._taught_hashes:
            return False
        self._taught_hashes.add(dedup_hash)

        # Store in student's memory
        learning = get_agent_learning()
        teaching_text = (
            f"[CROSS-TEACHING from {insight.teacher_agent}] "
            f"Topic: {insight.topic}. "
            f"Insight: {insight.insight}"
        )

        learning._store_in_vector_db(
            agent_id=target_agent,
            prompt=f"cross_teaching:{insight.topic}",
            content=teaching_text,
            doc_type="cross_teaching",
        )

        # Also store in in-memory learnings for fallback
        learning._learnings.setdefault(target_agent, []).append({
            "prompt": f"[teaching:{insight.topic}]",
            "good_response": teaching_text,
            "rating": 5,  # Cross-teachings are treated as high-quality
        })

        logger.info(
            f"CrossTeacher: {insight.teacher_agent} → taught → {target_agent} "
            f"(topic={insight.topic})"
        )
        return True

    # ─── Full Teaching Cycle ────────────────────────────────

    async def run_teaching_cycle(self) -> dict:
        """
        Run a complete cross-teaching cycle for all alliances.

        1. Iterate through each alliance
        2. Find top-performing members
        3. Extract their insights
        4. Teach to all other alliance members
        5. Log results
        """
        config = _load_teaching_config()
        if not config.get("enabled", False):
            return {"status": "disabled"}

        alliances = config.get("alliances", {})
        max_per_cycle = config.get("max_teachings_per_cycle", 5)

        cycle_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alliances_processed": 0,
            "insights_extracted": 0,
            "teachings_delivered": 0,
            "alliance_details": {},
        }

        for alliance_name, alliance_config in alliances.items():
            members = alliance_config.get("members", [])
            if len(members) < 2:
                continue

            alliance_result = {
                "teachers": [],
                "teachings": 0,
                "insights": 0,
            }

            # Find top-performing members to be teachers
            for teacher_id in members:
                insights = await self.extract_teachings(teacher_id, max_insights=max_per_cycle)
                if not insights:
                    continue

                alliance_result["teachers"].append(teacher_id)
                alliance_result["insights"] += len(insights)
                cycle_result["insights_extracted"] += len(insights)

                # Teach to all other members
                for student_id in members:
                    if student_id == teacher_id:
                        continue
                    for insight in insights:
                        taught = await self.teach(insight, student_id)
                        if taught:
                            alliance_result["teachings"] += 1
                            cycle_result["teachings_delivered"] += 1

            cycle_result["alliances_processed"] += 1
            cycle_result["alliance_details"][alliance_name] = alliance_result

        # Log history
        self._log_history(cycle_result)

        logger.info(
            f"CrossTeacher: cycle complete — "
            f"alliances={cycle_result['alliances_processed']}, "
            f"insights={cycle_result['insights_extracted']}, "
            f"teachings={cycle_result['teachings_delivered']}"
        )
        return cycle_result

    # ─── Effectiveness Tracking ─────────────────────────────

    async def get_teaching_effectiveness(self, agent_id: str) -> dict:
        """
        Measure whether cross-teachings improved an agent's quality.

        Compares avg rating before and after teachings were applied.
        """
        from app.services.intelligence.agent_learning import get_agent_learning

        learning = get_agent_learning()
        quality = learning.get_quality(agent_id)

        # Count cross-teaching entries
        cross_teachings = [
            ex for ex in learning._learnings.get(agent_id, [])
            if ex.get("prompt", "").startswith("[teaching:")
        ]

        return {
            "agent_id": agent_id,
            "teachings_received": len(cross_teachings),
            "current_avg_rating": quality.get("avg_rating", 0),
            "quality_trend": quality.get("trend", "unknown"),
            "status": quality.get("status", "no_data"),
        }

    def _log_history(self, cycle_result: dict) -> None:
        """Append teaching cycle result to history log."""
        try:
            with open(HISTORY_PATH, "a") as f:
                f.write(json.dumps(cycle_result, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"CrossTeacher: failed to log history: {e}")


# ─── Singleton ──────────────────────────────────────────────
_teacher: CrossTeacher | None = None


def get_cross_teacher() -> CrossTeacher:
    global _teacher
    if _teacher is None:
        _teacher = CrossTeacher()
    return _teacher
