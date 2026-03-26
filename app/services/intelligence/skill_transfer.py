"""
Skill Transfer Protocol — Portable knowledge packs between agents.

Software Factory Principle: Reusability + Knowledge Distillation.

Distills an agent's expertise into a portable "skill pack" that can
be injected into other agents. A skill pack contains:

1. Patterns — Output patterns that consistently get high ratings
2. Guardrails — Things to avoid (learned from corrections)
3. Format Rules — Preferred response formats
4. Domain Knowledge — Key facts the agent has learned

Architecture:
    Top Agent → create_skill_pack() → JSON Skill Pack
                                          ↓
    Target Agent ← apply_skill_pack() ← ChromaDB injection
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
SKILL_PACKS_DIR = Path("training/skill_packs")
SKILL_PACKS_DIR.mkdir(parents=True, exist_ok=True)


def _load_skill_config() -> dict:
    """Load skill transfer config from YAML."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}
    return config.get("skill_transfer", {})


class SkillPack:
    """A portable package of agent expertise."""

    def __init__(
        self,
        agent_id: str,
        domain: str,
        patterns: list[str],
        guardrails: list[str],
        format_rules: list[str],
        domain_knowledge: list[str],
    ):
        self.agent_id = agent_id
        self.domain = domain
        self.patterns = patterns
        self.guardrails = guardrails
        self.format_rules = format_rules
        self.domain_knowledge = domain_knowledge
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.pack_id = hashlib.md5(
            f"{agent_id}:{domain}:{self.created_at}".encode()
        ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "pack_id": self.pack_id,
            "agent_id": self.agent_id,
            "domain": self.domain,
            "patterns": self.patterns,
            "guardrails": self.guardrails,
            "format_rules": self.format_rules,
            "domain_knowledge": self.domain_knowledge,
            "created_at": self.created_at,
        }

    def to_context_string(self) -> str:
        """Convert to a string suitable for system prompt injection."""
        sections = []

        if self.patterns:
            sections.append("PATTERNS (follow these):\n" + "\n".join(f"• {p}" for p in self.patterns))

        if self.guardrails:
            sections.append("GUARDRAILS (avoid these):\n" + "\n".join(f"⚠ {g}" for g in self.guardrails))

        if self.format_rules:
            sections.append("FORMAT RULES:\n" + "\n".join(f"📋 {f}" for f in self.format_rules))

        if self.domain_knowledge:
            sections.append("DOMAIN KNOWLEDGE:\n" + "\n".join(f"💡 {d}" for d in self.domain_knowledge))

        return (
            f"\n--- SKILL PACK (from {self.agent_id}, domain: {self.domain}) ---\n"
            + "\n\n".join(sections)
            + "\n--- END SKILL PACK ---\n"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SkillPack":
        """Reconstruct a SkillPack from dict/JSON."""
        pack = cls(
            agent_id=data.get("agent_id", "unknown"),
            domain=data.get("domain", "general"),
            patterns=data.get("patterns", []),
            guardrails=data.get("guardrails", []),
            format_rules=data.get("format_rules", []),
            domain_knowledge=data.get("domain_knowledge", []),
        )
        pack.pack_id = data.get("pack_id", pack.pack_id)
        pack.created_at = data.get("created_at", pack.created_at)
        return pack


class SkillTransferProtocol:
    """
    Distills agent expertise into portable skill packs.

    Workflow:
    1. create_skill_pack(agent_id) — Analyze best responses → produce SkillPack
    2. apply_skill_pack(target, pack) — Inject into target agent's context
    3. get_skill_packs(domain) — List available packs for a domain
    """

    # ─── Create Skill Pack ──────────────────────────────────

    async def create_skill_pack(self, agent_id: str, domain: str = "general") -> SkillPack | None:
        """
        Analyze an agent's best responses and corrections to produce
        a structured skill document.

        Uses LLM to distill patterns into four categories:
        patterns, guardrails, format rules, domain knowledge.
        """
        from app.services.intelligence.agent_learning import get_agent_learning

        config = _load_skill_config()
        if not config.get("enabled", False):
            return None

        min_examples = config.get("min_examples_for_pack", 15)
        max_patterns = config.get("max_patterns_per_pack", 10)
        max_guardrails = config.get("max_guardrails_per_pack", 5)

        learning = get_agent_learning()
        good_examples = learning._learnings.get(agent_id, [])
        corrections = learning._corrections.get(agent_id, [])

        total_examples = len(good_examples) + len(corrections)
        if total_examples < min_examples:
            logger.debug(
                f"SkillTransfer: {agent_id} has {total_examples}/{min_examples} "
                f"examples, insufficient for skill pack"
            )
            return None

        # Use LLM to analyze and distill
        pack = await self._distill_skill_pack(
            agent_id, domain, good_examples, corrections, max_patterns, max_guardrails
        )

        if pack:
            # Save to disk
            self._save_pack(pack)
            logger.info(
                f"SkillTransfer: created skill pack for {agent_id} "
                f"({len(pack.patterns)} patterns, {len(pack.guardrails)} guardrails)"
            )

        return pack

    async def _distill_skill_pack(
        self,
        agent_id: str,
        domain: str,
        good_examples: list[dict],
        corrections: list[dict],
        max_patterns: int,
        max_guardrails: int,
    ) -> SkillPack | None:
        """Use LLM to distill expertise into structured categories."""
        from app.services.llm_service import get_llm_service

        llm = get_llm_service()

        # Build context
        examples_text = "\n".join(
            f"- Q: {ex['prompt'][:120]}\n  A (rated {ex.get('rating', 'high')}): {ex['good_response'][:250]}"
            for ex in good_examples[-25:]
        )

        corrections_text = "\n".join(
            f"- Q: {c['prompt'][:120]}\n  ❌ {c['bad_response'][:100]}\n  ✅ {c['correction'][:150]}"
            for c in corrections[-15:]
        )

        prompt = f"""Analyze these AI agent responses and create a structured "Skill Pack".

## Agent: {agent_id} (domain: {domain})

## Successful Responses:
{examples_text or "None"}

## Corrections (mistakes fixed):
{corrections_text or "None"}

## Distill into four categories:
1. **patterns** (max {max_patterns}): Output patterns that consistently work well
2. **guardrails** (max {max_guardrails}): Things to AVOID doing
3. **format_rules**: Preferred response structure/format
4. **domain_knowledge**: Key domain facts learned from interactions

Each item should be ONE concise, actionable sentence.

Return ONLY valid JSON:
{{
  "patterns": ["Always start financial analysis with a risk disclaimer", ...],
  "guardrails": ["Never provide specific investment advice without disclaimers", ...],
  "format_rules": ["Use JSON format with 'risk_score' field for analysis", ...],
  "domain_knowledge": ["Indian market hours are 9:15 AM to 3:30 PM IST", ...]
}}"""

        response = await llm.complete(
            prompt=prompt,
            system_prompt="You distill AI agent expertise into structured skill packs. Return only valid JSON.",
        )

        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                raw = json.loads(json_match.group())
                return SkillPack(
                    agent_id=agent_id,
                    domain=domain,
                    patterns=raw.get("patterns", [])[:max_patterns],
                    guardrails=raw.get("guardrails", [])[:max_guardrails],
                    format_rules=raw.get("format_rules", []),
                    domain_knowledge=raw.get("domain_knowledge", []),
                )
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"SkillTransfer: distillation parse failed: {e}")

        return None

    # ─── Apply Skill Pack ───────────────────────────────────

    async def apply_skill_pack(self, target_agent: str, pack: SkillPack) -> bool:
        """
        Inject a skill pack into a target agent's memory.

        The pack's content is stored in ChromaDB so the agent's
        future prompts can retrieve relevant skills.
        """
        from app.services.intelligence.agent_learning import get_agent_learning

        learning = get_agent_learning()
        context_string = pack.to_context_string()

        # Store in vector DB for semantic retrieval
        learning._store_in_vector_db(
            agent_id=target_agent,
            prompt=f"skill_pack:{pack.domain}:{pack.agent_id}",
            content=context_string,
            doc_type="skill_pack",
        )

        # Also store individual patterns for granular retrieval
        for i, pattern in enumerate(pack.patterns):
            learning._store_in_vector_db(
                agent_id=target_agent,
                prompt=f"pattern:{pack.domain}:{i}",
                content=f"[SKILL from {pack.agent_id}] Pattern: {pattern}",
                doc_type="skill_pattern",
            )

        for i, guardrail in enumerate(pack.guardrails):
            learning._store_in_vector_db(
                agent_id=target_agent,
                prompt=f"guardrail:{pack.domain}:{i}",
                content=f"[SKILL from {pack.agent_id}] Guardrail: {guardrail}",
                doc_type="skill_guardrail",
            )

        logger.info(
            f"SkillTransfer: applied pack from {pack.agent_id} → {target_agent} "
            f"({len(pack.patterns)} patterns, {len(pack.guardrails)} guardrails)"
        )
        return True

    # ─── List & Load ────────────────────────────────────────

    def get_skill_packs(self, domain: str | None = None) -> list[dict]:
        """List all available skill packs, optionally filtered by domain."""
        packs = []
        for pack_file in SKILL_PACKS_DIR.glob("*.json"):
            try:
                with open(pack_file) as f:
                    data = json.load(f)
                if domain and data.get("domain") != domain:
                    continue
                packs.append(data)
            except Exception:
                pass
        return sorted(packs, key=lambda x: x.get("created_at", ""), reverse=True)

    def load_skill_pack(self, pack_id: str) -> SkillPack | None:
        """Load a specific skill pack by ID."""
        pack_file = SKILL_PACKS_DIR / f"{pack_id}.json"
        if not pack_file.exists():
            return None
        try:
            with open(pack_file) as f:
                data = json.load(f)
            return SkillPack.from_dict(data)
        except Exception as e:
            logger.warning(f"SkillTransfer: failed to load pack {pack_id}: {e}")
            return None

    def _save_pack(self, pack: SkillPack) -> None:
        """Save skill pack to disk."""
        pack_file = SKILL_PACKS_DIR / f"{pack.pack_id}.json"
        try:
            with open(pack_file, "w") as f:
                json.dump(pack.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"SkillTransfer: failed to save pack: {e}")

    # ─── Full Cycle ─────────────────────────────────────────

    async def run_transfer_cycle(self) -> dict:
        """
        Full skill transfer cycle:
        1. Find top agents per alliance
        2. Create skill packs from top performers
        3. Apply packs to alliance members
        """
        config = _load_skill_config()
        if not config.get("enabled", False):
            return {"status": "disabled"}

        # Load alliances
        alliances_config_path = CONFIG_PATH
        if not alliances_config_path.exists():
            return {"status": "no_config"}
        with open(alliances_config_path) as f:
            full_config = yaml.safe_load(f) or {}
        alliances = full_config.get("cross_teaching", {}).get("alliances", {})

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "packs_created": 0,
            "packs_applied": 0,
            "alliance_details": {},
        }

        from app.services.intelligence.agent_learning import get_agent_learning
        learning = get_agent_learning()

        for alliance_name, alliance_config in alliances.items():
            members = alliance_config.get("members", [])
            alliance_result = {"packs_created": 0, "packs_applied": 0}

            for agent_id in members:
                quality = learning.get_quality(agent_id)
                if quality.get("avg_rating", 0) < 4.0:
                    continue

                # Create skill pack from top performer
                pack = await self.create_skill_pack(agent_id, domain=alliance_name)
                if not pack:
                    continue

                alliance_result["packs_created"] += 1
                result["packs_created"] += 1

                # Apply to other members
                for target_id in members:
                    if target_id == agent_id:
                        continue
                    applied = await self.apply_skill_pack(target_id, pack)
                    if applied:
                        alliance_result["packs_applied"] += 1
                        result["packs_applied"] += 1

            result["alliance_details"][alliance_name] = alliance_result

        logger.info(
            f"SkillTransfer: cycle complete — "
            f"created={result['packs_created']}, applied={result['packs_applied']}"
        )
        return result


# ─── Singleton ──────────────────────────────────────────────
_protocol: SkillTransferProtocol | None = None


def get_skill_transfer() -> SkillTransferProtocol:
    global _protocol
    if _protocol is None:
        _protocol = SkillTransferProtocol()
    return _protocol
