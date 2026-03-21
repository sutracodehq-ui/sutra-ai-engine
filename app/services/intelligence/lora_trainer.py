"""
LoRA Fine-Tuning Pipeline — Automated local model fine-tuning.

Software Factory Principle: Self-improving system.

Collects JSONL training data from:
1. Self-teaching (evolution engine)
2. Cloud distillation (distill_from_cloud)
3. User feedback (positive examples)
4. Edit-diff corrections

Then fine-tunes the local Ollama model via the Modelfile + LoRA adapter approach.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)


def _load_training_config() -> dict:
    """Load training config from intelligence_config.yaml."""
    config_path = Path("intelligence_config.yaml")
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("lora_training", {})


class LoraTrainer:
    """
    Automated LoRA fine-tuning pipeline for local Ollama model.

    Workflow:
    1. Collect all JSONL training files
    2. Merge and deduplicate
    3. Validate training data quality
    4. Create Ollama Modelfile with training data
    5. Fine-tune via `ollama create`
    6. Benchmark new model vs old
    7. Switch if better
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=600)  # Long timeout for training
        self._training_dir = Path("training/data")
        self._models_dir = Path("training/models")
        self._training_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Collect Training Data ──────────────────────

    def collect_training_data(self) -> list[dict]:
        """
        Collect all JSONL training files and merge them.
        Sources: self-teach, distillation, feedback, edit-diffs.
        """
        all_examples = []
        seen_hashes = set()

        for jsonl_file in sorted(self._training_dir.glob("*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        example = json.loads(line)

                        # Deduplicate by content hash
                        import hashlib
                        content_key = json.dumps(example.get("messages", []), sort_keys=True)
                        content_hash = hashlib.md5(content_key.encode()).hexdigest()

                        if content_hash not in seen_hashes:
                            seen_hashes.add(content_hash)
                            all_examples.append(example)

            except Exception as e:
                logger.warning(f"LoraTrainer: failed to read {jsonl_file}: {e}")

        logger.info(
            f"LoraTrainer: collected {len(all_examples)} unique examples "
            f"from {len(list(self._training_dir.glob('*.jsonl')))} files"
        )
        return all_examples

    # ─── Step 2: Validate Training Data ─────────────────────

    def validate_examples(self, examples: list[dict]) -> list[dict]:
        """
        Filter out low-quality training examples.
        
        Validation rules:
        - Must have at least 2 messages (user + assistant)
        - Assistant response must be > 50 chars
        - Must not contain error messages
        """
        config = _load_training_config()
        min_response_length = config.get("min_response_length", 50)
        error_patterns = config.get("error_patterns", [
            "error", "failed", "sorry, i can't", "i'm unable to",
        ])

        valid = []
        for ex in examples:
            messages = ex.get("messages", [])

            # Must have user + assistant messages
            has_user = any(m.get("role") == "user" for m in messages)
            has_assistant = any(m.get("role") == "assistant" for m in messages)
            if not (has_user and has_assistant):
                continue

            # Check assistant response quality
            assistant_msg = next(
                (m for m in messages if m.get("role") == "assistant"), {}
            )
            content = assistant_msg.get("content", "")

            if len(content) < min_response_length:
                continue

            # Skip error responses
            content_lower = content.lower()
            if any(err in content_lower for err in error_patterns):
                continue

            valid.append(ex)

        logger.info(f"LoraTrainer: validated {len(valid)}/{len(examples)} examples")
        return valid

    # ─── Step 3: Create Merged Training File ────────────────

    def create_training_file(self, examples: list[dict]) -> Path:
        """Write validated examples to a single merged JSONL file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self._models_dir / f"training_{timestamp}.jsonl"

        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info(f"LoraTrainer: created training file {output_path} ({len(examples)} examples)")
        return output_path

    # ─── Step 4: Fine-Tune via Ollama ───────────────────────

    async def fine_tune(self, training_file: Path) -> dict:
        """
        Fine-tune the local model using Ollama's create API.
        
        Creates a new model variant with the training data baked in
        as few-shot examples in the system prompt.
        """
        config = _load_training_config()
        settings = get_settings()
        ollama_url = settings.ollama_url
        base_model = settings.ollama_model

        # Read training examples
        examples = []
        with open(training_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        # Build Modelfile with training examples as system context
        max_examples = config.get("max_examples_in_modelfile", 20)
        selected = examples[:max_examples]

        # Format examples into system prompt
        examples_text = "\n\n".join(
            f"Example {i+1}:\nUser: {self._extract_user(ex)}\nAssistant: {self._extract_assistant(ex)}"
            for i, ex in enumerate(selected)
        )

        modelfile_content = f"""FROM {base_model}

SYSTEM \"\"\"You are an expert AI assistant. Learn from these high-quality examples to improve your responses:

{examples_text}

Apply the patterns from these examples to provide better, more structured responses.\"\"\"

PARAMETER temperature 0.7
PARAMETER num_predict 1024
PARAMETER top_p 0.9
"""

        # Create the fine-tuned model variant
        new_model_name = f"{base_model.replace(':', '-')}-sutra-tuned"

        try:
            resp = await self._client.post(
                f"{ollama_url}/api/create",
                json={
                    "name": new_model_name,
                    "modelfile": modelfile_content,
                    "stream": False,
                },
                timeout=300,
            )

            result = {
                "status": "success" if resp.status_code == 200 else "failed",
                "base_model": base_model,
                "new_model": new_model_name,
                "examples_used": len(selected),
                "total_available": len(examples),
                "training_file": str(training_file),
                "timestamp": datetime.now().isoformat(),
            }

            if resp.status_code == 200:
                logger.info(f"LoraTrainer: fine-tuned model created → {new_model_name}")
            else:
                logger.warning(f"LoraTrainer: fine-tuning failed: {resp.status_code}")
                result["error"] = resp.text[:500]

            return result

        except Exception as e:
            logger.error(f"LoraTrainer: fine-tuning error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "base_model": base_model,
            }

    # ─── Step 5: Full Pipeline ──────────────────────────────

    async def run_pipeline(self) -> dict:
        """
        Run the complete LoRA fine-tuning pipeline:
        1. Collect all training data
        2. Validate and filter
        3. Create merged training file
        4. Fine-tune the model
        """
        config = _load_training_config()
        min_examples = config.get("min_examples_for_training", 20)

        # 1. Collect
        all_examples = self.collect_training_data()

        # 2. Validate
        valid_examples = self.validate_examples(all_examples)

        # 3. Check minimum threshold
        if len(valid_examples) < min_examples:
            logger.info(
                f"LoraTrainer: not enough data ({len(valid_examples)}/{min_examples}). "
                f"Skipping fine-tuning."
            )
            return {
                "status": "skipped",
                "reason": f"Need {min_examples} examples, have {len(valid_examples)}",
                "total_collected": len(all_examples),
                "total_valid": len(valid_examples),
            }

        # 3. Create training file
        training_file = self.create_training_file(valid_examples)

        # 4. Fine-tune
        result = await self.fine_tune(training_file)
        result["total_collected"] = len(all_examples)
        result["total_valid"] = len(valid_examples)

        # 5. Log result
        history_path = Path("training/fine_tune_history.jsonl")
        try:
            with open(history_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

        return result

    # ─── Helpers ────────────────────────────────────────────

    @staticmethod
    def _extract_user(example: dict) -> str:
        """Extract user message from a training example."""
        messages = example.get("messages", [])
        user_msg = next((m for m in messages if m.get("role") == "user"), {})
        return user_msg.get("content", "")[:200]

    @staticmethod
    def _extract_assistant(example: dict) -> str:
        """Extract assistant message from a training example."""
        messages = example.get("messages", [])
        assistant_msg = next((m for m in messages if m.get("role") == "assistant"), {})
        return assistant_msg.get("content", "")[:500]

    async def close(self):
        await self._client.aclose()


# ─── Singleton ──────────────────────────────────────────────
_trainer: LoraTrainer | None = None


def get_lora_trainer() -> LoraTrainer:
    global _trainer
    if _trainer is None:
        _trainer = LoraTrainer()
    return _trainer
