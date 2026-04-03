"""
Agent Evaluator — Per-agent benchmark and quality scoring.

Evaluates fine-tuned models against gold-standard test sets and
the base model to ensure quality improvements before deployment.

Scoring dimensions:
1. Schema compliance — are all required JSON fields present?
2. Response quality — length, coherence, keyword coverage
3. Latency — response time in ms
4. Domain specificity — domain-relevant terms present
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path("training/benchmarks")
GOLD_DIR = BENCHMARKS_DIR / "gold"
RESULTS_DIR = BENCHMARKS_DIR / "results"
AGENT_CONFIG_DIR = Path("agent_config")


@dataclass
class BenchmarkResult:
    """Result of evaluating one test case."""
    agent_type: str
    prompt: str
    model: str
    driver: str
    response: str
    latency_ms: float
    schema_compliance: float  # 0.0-1.0
    keyword_score: float      # 0.0-1.0
    quality_score: float      # 0.0-10.0
    fields_present: list[str] = field(default_factory=list)
    fields_missing: list[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "prompt": self.prompt[:200],
            "model": self.model,
            "driver": self.driver,
            "latency_ms": self.latency_ms,
            "schema_compliance": self.schema_compliance,
            "keyword_score": self.keyword_score,
            "quality_score": self.quality_score,
            "fields_missing": self.fields_missing,
            "passed": self.passed,
        }


@dataclass
class AgentBenchmarkReport:
    """Aggregated benchmark results for one agent."""
    agent_type: str
    model: str
    total_tests: int
    passed: int
    failed: int
    avg_quality: float
    avg_schema_compliance: float
    avg_keyword_score: float
    avg_latency_ms: float
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_tests if self.total_tests > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "model": self.model,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 3),
            "avg_quality": round(self.avg_quality, 2),
            "avg_schema_compliance": round(self.avg_schema_compliance, 3),
            "avg_keyword_score": round(self.avg_keyword_score, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class AgentEvaluator:
    """
    Evaluates agent models against gold-standard test sets.

    Test sets are stored in training/benchmarks/gold/{agent_type}.jsonl
    Each line: {"prompt": "...", "expected_fields": [...], "keywords": [...], "min_quality": 7}
    """

    def __init__(self):
        GOLD_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_test_set(self, agent_type: str) -> list[dict]:
        """Load gold-standard test cases for an agent."""
        path = GOLD_DIR / f"{agent_type}.jsonl"
        if not path.exists():
            return []
        tests = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    tests.append(json.loads(line))
        return tests

    def generate_default_tests(self, agent_type: str, count: int = 10) -> list[dict]:
        """
        Auto-generate test cases from agent config when no gold set exists.
        Uses the agent's domain, capabilities, and response_schema to create tests.
        """
        config_path = AGENT_CONFIG_DIR / f"{agent_type}.yaml"
        if not config_path.exists():
            return []

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        domain = config.get("domain", "general")
        name = config.get("name", agent_type)
        capabilities = config.get("capabilities", [])

        schema = config.get("response_schema", {})
        if isinstance(schema, dict):
            expected_fields = schema.get("fields", [])
        elif isinstance(schema, list):
            expected_fields = schema
        else:
            expected_fields = []

        # Generate test prompts from capabilities
        tests = []
        for i, cap in enumerate(capabilities[:count]):
            tests.append({
                "prompt": f"As a {name}, demonstrate: {cap}",
                "expected_fields": expected_fields,
                "keywords": [],
                "min_quality": 6,
            })

        # Add domain-specific tests
        domain_prompts = {
            "finance": [
                "Analyze AAPL stock performance and give buy/sell recommendation",
                "Calculate ROI for a $50,000 investment returning $72,000 in 3 years",
            ],
            "edtech": [
                "Generate a quiz on photosynthesis with 5 MCQs and 3 true/false questions",
                "Create study notes on Newton's Laws of Motion for Class 11",
            ],
            "marketing": [
                "Write an SEO-optimized blog outline about digital marketing trends",
                "Create a social media content calendar for a fitness brand",
            ],
            "health": [
                "Explain common symptoms of vitamin D deficiency",
                "Create a balanced diet plan for a diabetic patient",
            ],
        }

        for prompt in domain_prompts.get(domain, [])[:count - len(tests)]:
            tests.append({
                "prompt": prompt,
                "expected_fields": expected_fields,
                "keywords": [],
                "min_quality": 6,
            })

        return tests[:count]

    def evaluate(
        self,
        agent_type: str,
        response: str,
        expected_fields: list[str] = None,
        keywords: list[str] = None,
        min_quality: float = 6.0,
    ) -> dict:
        """Public API to evaluate a single response."""
        return self.score_response(response, expected_fields or [], keywords or [], min_quality)

    def score_response(
        self,
        response_text: str,
        expected_fields: list[str],
        keywords: list[str],
        min_quality: float = 6.0,
    ) -> dict:
        """
        Score a response on multiple dimensions.

        Returns:
            {
                "schema_compliance": float,
                "keyword_score": float,
                "quality_score": float,
                "fields_present": [...],
                "fields_missing": [...],
                "passed": bool,
            }
        """
        # Schema compliance
        fields_present = []
        fields_missing = list(expected_fields)

        if expected_fields:
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, dict):
                    fields_present = [f for f in expected_fields if f in parsed]
                    fields_missing = [f for f in expected_fields if f not in parsed]
            except (json.JSONDecodeError, TypeError):
                # Try extracting from markdown
                import re
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, dict):
                            fields_present = [f for f in expected_fields if f in parsed]
                            fields_missing = [f for f in expected_fields if f not in parsed]
                    except (json.JSONDecodeError, TypeError):
                        pass

        schema_compliance = len(fields_present) / len(expected_fields) if expected_fields else 1.0

        # Keyword coverage
        if keywords:
            text_lower = response_text.lower()
            hits = sum(1 for kw in keywords if kw.lower() in text_lower)
            keyword_score = hits / len(keywords)
        else:
            keyword_score = 1.0

        # Quality score (composite)
        length_score = min(len(response_text) / 200, 1.0)  # Penalize very short
        structure_score = 1.0 if (response_text.strip().startswith("{") or len(response_text) > 100) else 0.5

        quality_score = (
            schema_compliance * 4.0 +     # 40% weight on schema
            keyword_score * 2.0 +          # 20% weight on keywords
            length_score * 2.0 +           # 20% weight on length
            structure_score * 2.0           # 20% weight on structure
        )

        return {
            "schema_compliance": schema_compliance,
            "keyword_score": keyword_score,
            "quality_score": quality_score,
            "fields_present": fields_present,
            "fields_missing": fields_missing,
            "passed": quality_score >= min_quality,
        }

    async def evaluate_agent(
        self,
        agent_type: str,
        driver: str = "ollama",
        model: str | None = None,
    ) -> AgentBenchmarkReport:
        """
        Run full benchmark suite for an agent.

        Uses gold-standard tests if available, otherwise auto-generates tests.
        """
        from app.services.llm_service import get_llm_service

        # Load or generate tests
        tests = self.load_test_set(agent_type)
        if not tests:
            tests = self.generate_default_tests(agent_type)
            if not tests:
                return AgentBenchmarkReport(
                    agent_type=agent_type, model=model or "unknown",
                    total_tests=0, passed=0, failed=0,
                    avg_quality=0, avg_schema_compliance=0,
                    avg_keyword_score=0, avg_latency_ms=0,
                )

        # Load agent system prompt
        config_path = AGENT_CONFIG_DIR / f"{agent_type}.yaml"
        system_prompt = f"You are a {agent_type} agent."
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            system_prompt = config.get("system_prompt", system_prompt)

            # Inject schema instructions
            schema = config.get("response_schema", {})
            if isinstance(schema, dict) and schema.get("format") == "json":
                fields = schema.get("fields", [])
                fields_str = ", ".join(f'"{f}"' for f in fields)
                system_prompt += f"\n\nRespond with valid JSON only. Required keys: {fields_str}"

        llm = get_llm_service()
        results = []

        for test in tests:
            prompt = test["prompt"]
            expected_fields = test.get("expected_fields", [])
            keywords = test.get("keywords", [])
            min_quality = test.get("min_quality", 6.0)

            try:
                start = time.monotonic()
                response = await llm.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    driver=driver,
                    model=model,
                )
                latency_ms = (time.monotonic() - start) * 1000

                scores = self.score_response(
                    response.content, expected_fields, keywords, min_quality,
                )

                result = BenchmarkResult(
                    agent_type=agent_type,
                    prompt=prompt,
                    model=response.model or model or "unknown",
                    driver=response.driver or driver,
                    response=response.content[:500],
                    latency_ms=latency_ms,
                    **scores,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Evaluator: {agent_type} test failed: {e}")
                results.append(BenchmarkResult(
                    agent_type=agent_type, prompt=prompt,
                    model=model or "unknown", driver=driver,
                    response="", latency_ms=0,
                    schema_compliance=0, keyword_score=0,
                    quality_score=0, passed=False,
                ))

        # Aggregate
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_quality = sum(r.quality_score for r in results) / total if total else 0
        avg_schema = sum(r.schema_compliance for r in results) / total if total else 0
        avg_keyword = sum(r.keyword_score for r in results) / total if total else 0
        avg_latency = sum(r.latency_ms for r in results) / total if total else 0

        report = AgentBenchmarkReport(
            agent_type=agent_type,
            model=model or results[0].model if results else "unknown",
            total_tests=total,
            passed=passed,
            failed=total - passed,
            avg_quality=avg_quality,
            avg_schema_compliance=avg_schema,
            avg_keyword_score=avg_keyword,
            avg_latency_ms=avg_latency,
            results=results,
        )

        # Save results
        self._save_report(report)

        return report

    async def compare_models(
        self,
        agent_type: str,
        base_driver: str = "ollama",
        base_model: str | None = None,
        new_driver: str = "ollama",
        new_model: str | None = None,
    ) -> dict:
        """
        Compare two models on the same benchmark suite.

        Returns comparison dict with per-dimension deltas.
        """
        base_report = await self.evaluate_agent(agent_type, base_driver, base_model)
        new_report = await self.evaluate_agent(agent_type, new_driver, new_model)

        return {
            "agent_type": agent_type,
            "base": base_report.to_dict(),
            "new": new_report.to_dict(),
            "delta": {
                "quality": round(new_report.avg_quality - base_report.avg_quality, 2),
                "schema_compliance": round(new_report.avg_schema_compliance - base_report.avg_schema_compliance, 3),
                "keyword_score": round(new_report.avg_keyword_score - base_report.avg_keyword_score, 3),
                "latency_ms": round(new_report.avg_latency_ms - base_report.avg_latency_ms, 1),
                "pass_rate": round(new_report.pass_rate - base_report.pass_rate, 3),
            },
            "recommendation": (
                "deploy" if new_report.avg_quality >= base_report.avg_quality - 0.5
                else "reject"
            ),
        }

    def _save_report(self, report: AgentBenchmarkReport) -> None:
        """Save benchmark report to results directory."""
        from datetime import datetime
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        path = RESULTS_DIR / f"{report.agent_type}_{report.model.replace(':', '-')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            data = report.to_dict()
            data["results"] = [r.to_dict() for r in report.results]
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluator: saved report → {path}")


# ─── Singleton ────────────────────────────────────────────────
_evaluator: AgentEvaluator | None = None


def get_agent_evaluator() -> AgentEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = AgentEvaluator()
    return _evaluator
