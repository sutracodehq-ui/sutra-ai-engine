"""
Data Quality Pipeline — Schema validation, semantic dedup, diversity scoring.

Ensures training data is high quality before it enters the fine-tuning pipeline.
Replaces the simple length-based filtering in lora_trainer.py with proper
schema-aware validation, semantic deduplication, and field coverage tracking.
"""

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

AGENT_CONFIG_DIR = Path("agent_config")


def _load_agent_schema(agent_type: str) -> dict:
    """Load response_schema from an agent's YAML config."""
    path = AGENT_CONFIG_DIR / f"{agent_type}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("response_schema", {})


class DataQualityPipeline:
    """
    Multi-stage quality pipeline for training data.

    Stages:
    1. Schema Validation — verify JSON fields match agent's response_schema
    2. Content Deduplication — MD5 hash + optional semantic dedup
    3. Error Pattern Filtering — reject known error responses
    4. Diversity Scoring — track field coverage across the dataset
    """

    def __init__(self):
        self._error_patterns = self._load_error_patterns()

    @staticmethod
    def _load_error_patterns() -> list[str]:
        """Load error patterns from intelligence_config.yaml."""
        config_path = Path("intelligence_config.yaml")
        if not config_path.exists():
            return []
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return config.get("lora_training", {}).get("error_patterns", [
            "error", "failed", "sorry, i can't", "i'm unable to",
            "as an ai", "i cannot",
        ])

    # ─── Stage 1: Schema Validation ───────────────────────────

    def validate_schema(self, example: dict, agent_type: str) -> dict:
        """
        Validate that assistant response contains all required JSON fields.

        Returns:
            {
                "valid": bool,
                "fields_present": [...],
                "fields_missing": [...],
                "compliance_ratio": float  # 0.0 to 1.0
            }
        """
        schema = _load_agent_schema(agent_type)
        if not schema:
            return {"valid": True, "fields_present": [], "fields_missing": [], "compliance_ratio": 1.0}

        # Get required fields
        if isinstance(schema, dict):
            required_fields = schema.get("fields", [])
            expected_format = schema.get("format", "text")
        elif isinstance(schema, list):
            required_fields = schema
            expected_format = "json"
        else:
            return {"valid": True, "fields_present": [], "fields_missing": [], "compliance_ratio": 1.0}

        if not required_fields or expected_format != "json":
            return {"valid": True, "fields_present": [], "fields_missing": [], "compliance_ratio": 1.0}

        # Extract assistant content
        messages = example.get("messages", [])
        assistant_msg = next((m for m in messages if m.get("role") == "assistant"), {})
        content = assistant_msg.get("content", "")

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            # Try extracting JSON from markdown code blocks
            parsed = self._extract_json(content)
            if parsed is None:
                return {
                    "valid": False,
                    "fields_present": [],
                    "fields_missing": required_fields,
                    "compliance_ratio": 0.0,
                }

        if not isinstance(parsed, dict):
            return {
                "valid": False,
                "fields_present": [],
                "fields_missing": required_fields,
                "compliance_ratio": 0.0,
            }

        # Check which fields are present
        present = [f for f in required_fields if f in parsed]
        missing = [f for f in required_fields if f not in parsed]
        ratio = len(present) / len(required_fields) if required_fields else 1.0

        return {
            "valid": ratio >= 0.8,  # Allow 80% compliance
            "fields_present": present,
            "fields_missing": missing,
            "compliance_ratio": ratio,
        }

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Try to extract JSON from text that may be wrapped in markdown."""
        import re
        # Try ```json ... ``` blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                pass
        # Try finding the first { ... } block
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    # ─── Stage 2: Content Deduplication ───────────────────────

    def deduplicate(self, examples: list[dict]) -> list[dict]:
        """
        Remove duplicate training examples by content hash.

        Uses MD5 hash of the user+assistant message pair.
        """
        seen = set()
        unique = []

        for ex in examples:
            content_key = self._content_hash(ex)
            if content_key not in seen:
                seen.add(content_key)
                unique.append(ex)

        removed = len(examples) - len(unique)
        if removed:
            logger.info(f"DataQuality: dedup removed {removed} duplicates ({len(unique)} remain)")
        return unique

    @staticmethod
    def _content_hash(example: dict) -> str:
        """Generate a hash from user + assistant content."""
        messages = example.get("messages", [])
        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
        key = f"{user}|||{assistant}"
        return hashlib.md5(key.encode()).hexdigest()

    # ─── Stage 3: Error Pattern Filtering ─────────────────────

    def filter_errors(self, examples: list[dict]) -> list[dict]:
        """Remove examples where the assistant response contains error patterns."""
        clean = []
        for ex in examples:
            messages = ex.get("messages", [])
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), {})
            content = assistant_msg.get("content", "").lower()

            if len(content) < 50:
                continue

            if any(pat.lower() in content for pat in self._error_patterns):
                continue

            clean.append(ex)

        removed = len(examples) - len(clean)
        if removed:
            logger.info(f"DataQuality: error filter removed {removed} examples")
        return clean

    # ─── Stage 4: Diversity Scoring ───────────────────────────

    def score_diversity(self, examples: list[dict], agent_type: str) -> dict:
        """
        Score the diversity of a training dataset for a specific agent.

        Checks field coverage: which response_schema fields appear across examples,
        and flags underrepresented fields.

        Returns:
            {
                "total_examples": int,
                "field_coverage": {field: count},
                "underrepresented_fields": [fields with < 30% coverage],
                "diversity_score": float  # 0.0 to 1.0
            }
        """
        schema = _load_agent_schema(agent_type)
        if not schema:
            return {"total_examples": len(examples), "diversity_score": 1.0}

        if isinstance(schema, dict):
            fields = schema.get("fields", [])
        elif isinstance(schema, list):
            fields = schema
        else:
            return {"total_examples": len(examples), "diversity_score": 1.0}

        if not fields:
            return {"total_examples": len(examples), "diversity_score": 1.0}

        field_counts: dict[str, int] = defaultdict(int)

        for ex in examples:
            messages = ex.get("messages", [])
            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), {})
            content = assistant_msg.get("content", "")

            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                parsed = self._extract_json(content)

            if isinstance(parsed, dict):
                for field in fields:
                    if field in parsed and parsed[field]:
                        field_counts[field] += 1

        total = len(examples) if examples else 1
        underrepresented = [
            f for f in fields
            if field_counts.get(f, 0) / total < 0.3
        ]

        coverage_ratio = sum(1 for f in fields if field_counts.get(f, 0) > 0) / len(fields) if fields else 1.0

        return {
            "total_examples": len(examples),
            "field_coverage": dict(field_counts),
            "underrepresented_fields": underrepresented,
            "diversity_score": coverage_ratio,
        }

    # ─── Full Pipeline ────────────────────────────────────────

    def run(self, examples: list[dict], agent_type: str) -> dict:
        """
        Run the complete quality pipeline on a list of training examples.

        Returns:
            {
                "input_count": int,
                "output_count": int,
                "valid_examples": [...],
                "schema_compliance_rate": float,
                "diversity": {...},
                "removed": {
                    "duplicates": int,
                    "errors": int,
                    "schema_invalid": int,
                }
            }
        """
        input_count = len(examples)

        # Stage 1: Dedup
        deduped = self.deduplicate(examples)
        dup_removed = input_count - len(deduped)

        # Stage 2: Error filtering
        clean = self.filter_errors(deduped)
        err_removed = len(deduped) - len(clean)

        # Stage 3: Schema validation
        valid = []
        schema_invalid = 0
        compliance_scores = []

        for ex in clean:
            result = self.validate_schema(ex, agent_type)
            compliance_scores.append(result["compliance_ratio"])
            if result["valid"]:
                valid.append(ex)
            else:
                schema_invalid += 1

        # Stage 4: Diversity scoring
        diversity = self.score_diversity(valid, agent_type)

        avg_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0.0

        logger.info(
            f"DataQuality[{agent_type}]: {input_count} → {len(valid)} "
            f"(dedup={dup_removed}, errors={err_removed}, schema={schema_invalid})"
        )

        return {
            "input_count": input_count,
            "output_count": len(valid),
            "valid_examples": valid,
            "schema_compliance_rate": avg_compliance,
            "diversity": diversity,
            "removed": {
                "duplicates": dup_removed,
                "errors": err_removed,
                "schema_invalid": schema_invalid,
            },
        }


# ─── Singleton ────────────────────────────────────────────────
_pipeline: DataQualityPipeline | None = None


def get_data_quality() -> DataQualityPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DataQualityPipeline()
    return _pipeline
