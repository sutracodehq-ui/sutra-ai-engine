"""
Live Knowledge Service — Search-driven model improvement.

Pipeline:
1. KnowledgeSeeker: Proactively search for field updates (Tavily/Serp).
2. KnowledgeDistiller: Distill search results into training examples (JSONL).
3. LoraTrainer: Continuous fine-tuning in the background.

Software Factory Principle: Autonomous, non-blocking self-improvement.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

import httpx
import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


class KnowledgeSeeker:
    """
    Search-driven knowledge acquisition.
    Uses Tavily to find latest developments in specific domains.
    """

    def __init__(self):
        self._settings = get_settings()
        self._client = httpx.AsyncClient(timeout=30)

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web using Tavily."""
        if not self._settings.tavily_api_key:
            logger.warning("KnowledgeSeeker: Tavily API key missing, skipping search")
            return []

        try:
            resp = await self._client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._settings.tavily_api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                    "include_answer": True,
                    "include_raw_content": False,
                },
            )
            
            if resp.status_code != 200:
                logger.error(f"KnowledgeSeeker: Tavily API error ({resp.status_code}): {resp.text}")
                return []

            data = resp.json()
            results = data.get("results", [])
            logger.info(f"KnowledgeSeeker: found {len(results)} results for '{query}'")
            return results

        except Exception as e:
            logger.error(f"KnowledgeSeeker: search failed: {e}")
            return []

    async def close(self):
        await self._client.aclose()


class KnowledgeDistiller:
    """
    Converts raw search results into high-quality training pairs.
    Uses a frontier model (Gemini/GPT-4o) for distillation.
    """

    def __init__(self):
        self._settings = get_settings()
        from app.services.llm_service import get_llm_service
        self._llm = get_llm_service()

    async def distill(self, domain: str, search_results: list[dict]) -> list[dict]:
        """
        Process search results and generate training examples.
        Each example is a {messages: [...]} dict for JSONL.
        """
        if not search_results:
            return []

        examples = []
        # Group content for the teacher model
        context_text = "\n\n".join([
            f"Source: {r.get('url')}\nTitle: {r.get('title')}\nContent: {r.get('content')}"
            for r in search_results
        ])

        system_prompt = (
            f"You are a master {domain} expert and teacher. "
            "Your task is to take the provided research context and create 3-5 HIGH-QUALITY "
            "Instruction-following examples for a training dataset. "
            "Each example must be a challenging real-world question and a detailed, "
            "expert-level answer based on the LATEST information provided. "
            "Output ONLY a valid JSON list of objects: "
            '[{"user": "question", "assistant": "answer"}, ...]'
        )

        try:
            prompt = f"Produce a training dataset from this context:\n\n{context_text}"
            
            # Use a frontier cloud model for distillation (never local qwen/llama)
            response = await self._llm.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                model="gemini-2.0-flash", # Fast, high context, great at distillation
                temperature=0.4
            )

            # Robust JSON extraction
            content = response.content.strip()
            
            # Use Brain's internal cleaner if content is messy
            clean_json = content
            if "```" in content:
                import re
                m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
                if m:
                    clean_json = m.group(1).strip()
            
            try:
                raw_examples = json.loads(clean_json)
            except json.JSONDecodeError:
                # Try finding any array in the text
                import re
                m = re.search(r"\[\s*\{.*\}\s*\]", clean_json, re.DOTALL)
                if m:
                    raw_examples = json.loads(m.group(0))
                else:
                    logger.error(f"KnowledgeDistiller: failed to find JSON array in: {content[:100]}...")
                    return []
            
            if not isinstance(raw_examples, list):
                logger.error(f"KnowledgeDistiller: expected list, got {type(raw_examples)}")
                return []
            
            for ex in raw_examples:
                examples.append({
                    "messages": [
                        {"role": "system", "content": f"You are a professional {domain} consultant."},
                        {"role": "user", "content": ex["user"]},
                        {"role": "assistant", "content": ex["assistant"]},
                    ],
                    "source": "live_knowledge",
                    "domain": domain,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            logger.info(f"KnowledgeDistiller: distilled {len(examples)} examples for {domain}")
            return examples

        except Exception as e:
            logger.error(f"KnowledgeDistiller: distillation failed for {domain}: {e}")
            return []


class LiveKnowledgeCycle:
    """
    Orchestrates the Seeker -> Distiller -> Dataset pipeline.
    This should be called by a Celery worker.
    """

    def __init__(self):
        self._seeker = KnowledgeSeeker()
        self._distiller = KnowledgeDistiller()
        self._training_dir = Path("training/data")
        self._training_dir.mkdir(parents=True, exist_ok=True)

    async def run_cycle(self, domain: str, query: str):
        """Execute one knowledge acquisition cycle."""
        logger.info(f"LiveKnowledge: starting cycle for {domain} ('{query}')")
        
        # 1. Search
        results = await self._seeker.search(query)
        if not results:
            return 0

        # 2. Distill
        examples = await self._distiller.distill(domain, results)
        if not examples:
            return 0

        # 3. Save to JSONL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_knowledge_{domain}_{timestamp}.jsonl"
        output_path = self._training_dir / filename

        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info(f"LiveKnowledge: cycle complete. Saved {len(examples)} examples to {output_path}")
        return len(examples)

    async def close(self):
        await self._seeker.close()


# ─── Singleton ──────────────────────────────────────────────
_cycle: LiveKnowledgeCycle | None = None

def get_live_knowledge_cycle() -> LiveKnowledgeCycle:
    global _cycle
    if _cycle is None:
        _cycle = LiveKnowledgeCycle()
    return _cycle
