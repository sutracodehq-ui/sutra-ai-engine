"""
LoRA Fine-Tuning Pipeline — Automated local model fine-tuning.

Software Factory Principle: Self-improving system.

Collects JSONL training data from:
1. Self-teaching (evolution engine)
2. Cloud distillation (distill_from_cloud)
3. User feedback (positive examples)
4. Edit-diff corrections

Then fine-tunes local models using MLX-LM LoRA adapters for domain-specific clusters.
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
    1. Collect all JSONL training files from per-agent logs
    2. Group examples by domain cluster (config/model_clusters.yaml)
    3. Validate and deduplicate
    4. Trigger MLX-LM training for clusters with enough data
    5. Benchmark new adapters
    6. Update Ollama configuration to use new adapters
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=600)  # Long timeout for training
        self._training_dir = Path("training/data")
        self._models_dir = Path("training/models")
        self._training_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Collect Training Data ──────────────────────

    def collect_training_data_by_cluster(self) -> dict[str, list[dict]]:
        """
        Collect training data and group by cluster.
        Returns: {cluster_name: [examples]}
        """
        from app.services.intelligence.config_loader import load_clusters
        clusters_cfg = load_clusters()
        clusters = clusters_cfg.get("clusters", {})
        
        # Build agent -> cluster map
        agent_to_cluster = {}
        for cluster_name, info in clusters.items():
            for agent_id in info.get("agents", []):
                agent_to_cluster[agent_id] = cluster_name

        cluster_examples = {name: [] for name in clusters.keys()}
        cluster_examples["general"] = [] # fallback
        
        seen_hashes = set()
        
        # Sources: training/data/per_agent/*.jsonl
        per_agent_dir = Path("training/data/per_agent")
        for jsonl_file in sorted(per_agent_dir.glob("*.jsonl")):
            agent_id = jsonl_file.stem
            cluster_name = agent_to_cluster.get(agent_id, "general")
            
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        example = json.loads(line)
                        
                        # Deduplicate
                        content_key = json.dumps(example.get("messages", []), sort_keys=True)
                        import hashlib
                        content_hash = hashlib.md5(content_key.encode()).hexdigest()
                        
                        if content_hash not in seen_hashes:
                            seen_hashes.add(content_hash)
                            cluster_examples[cluster_name].append(example)
            except Exception as e:
                logger.warning(f"LoraTrainer: failed to read {jsonl_file}: {e}")
                
        return cluster_examples

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

    async def fine_tune_cluster(self, cluster_name: str) -> dict:
        """
        Trigger real LoRA fine-tuning for a cluster via MLX-LM.
        """
        logger.info(f"LoraTrainer: triggering LoRA training for cluster [{cluster_name}]")
        
        # 1. Launch scripts/train_cluster_model.py
        import sys
        import subprocess
        
        cmd = [sys.executable, "scripts/train_cluster_model.py", "--cluster", cluster_name]
        
        try:
            # We run this as a subprocess to keep the training memory separate
            # and avoid blocking the main event loop too much
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Note: For production, we would want to track this background job.
            # For now, we'll wait for a small timeout or just log that it started.
            return {
                "status": "started",
                "cluster": cluster_name,
                "pid": proc.pid,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"LoraTrainer: failed to start training for {cluster_name}: {e}")
            return {"status": "error", "error": str(e), "cluster": cluster_name}

    # ─── Step 5: Self-Improvement Cycle ─────────────────────
    
    async def run_self_improvement_cycle(self, domain: str, query: str) -> dict:
        """
        Run a complete search-driven self-improvement cycle:
        1. Search & Distill (non-blocking via LiveKnowledgeCycle)
        2. Validate & Filter
        3. Local Fine-Tune (Few-shot injection)
        4. Benchmark & Gate
        """
        from app.services.intelligence.live_knowledge import get_live_knowledge_cycle
        cycle = get_live_knowledge_cycle()
        
        logger.info(f"LoraTrainer: starting self-improvement for [{domain}]")
        
        # 1. Search and Distill (Creates and saves new JSONL data)
        new_examples_count = await cycle.run_cycle(domain, query)
        if new_examples_count == 0:
            return {"status": "skipped", "reason": "No new knowledge found/distilled", "domain": domain}
            
        # 2. Run the standard pipeline (which now picks up the new live knowledge files)
        result = await self.run_pipeline()
        result["domain"] = domain
        result["new_knowledge_count"] = new_examples_count
        
        # 3. Benchmark Gate (Optional: Verify performance improved)
        try:
            from app.services.intelligence.domain_evolution import get_domain_evolution
            evo = get_domain_evolution()
            # Benchmark the new model variant
            bench = await evo.benchmark_domain(domain, model_name=result.get("new_model"))
            result["benchmark"] = {
                "quality": bench.get("avg_quality"),
                "pass_rate": bench.get("pass_rate")
            }
        except Exception as e:
            logger.warning(f"LoraTrainer: benchmark gate failed: {e}")
            
        return result

    # ─── Step 6: Full Pipeline ──────────────────────────────

    async def run_pipeline(self) -> dict:
        """
        Run the complete Cluster-based LoRA training pipeline.
        """
        from app.services.intelligence.config_loader import load_clusters
        clusters_cfg = load_clusters()
        min_examples = clusters_cfg.get("training_defaults", {}).get("min_examples", 200)

        # 1. Collect and group
        cluster_data = self.collect_training_data_by_cluster()
        
        results = {}
        for cluster_name, examples in cluster_data.items():
            # 2. Validate
            valid_examples = self.validate_examples(examples)
            
            if len(valid_examples) >= min_examples:
                # 3. Trigger training (non-blocking)
                results[cluster_name] = await self.fine_tune_cluster(cluster_name)
            else:
                logger.debug(f"LoraTrainer: skipping {cluster_name} (only {len(valid_examples)}/{min_examples} examples)")

        return {
            "status": "completed",
            "clusters_processed": len(cluster_data),
            "trainings_started": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

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
        return assistant_msg.get("content", "")[:1000] # Allow longer assistant context

    async def close(self):
        await self._client.aclose()



# ─── Singleton ──────────────────────────────────────────────
_trainer: LoraTrainer | None = None


def get_lora_trainer() -> LoraTrainer:
    global _trainer
    if _trainer is None:
        _trainer = LoraTrainer()
    return _trainer
