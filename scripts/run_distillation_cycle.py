"""
Distillation Cycle Orchestrator — Domain-aware distillation per cluster.

Orchestrates the full distillation pipeline:
1. For each cluster, load all agents
2. Generate domain-specific synthetic prompts
3. Send to the cluster's designated teacher model
4. Validate responses against agent schemas
5. Store as per-agent JSONL files
6. Track progress and trigger training when threshold is met

Usage:
    python scripts/run_distillation_cycle.py --cluster marketing --samples 20
    python scripts/run_distillation_cycle.py --all --samples 10
    python scripts/run_distillation_cycle.py --agent quiz_generator --samples 30
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLUSTER_CONFIG = Path("config/model_clusters.yaml")
AGENT_CONFIG_DIR = Path("agent_config")
OUTPUT_DIR = Path("training/data/per_agent")


def load_config() -> dict:
    """Load cluster config."""
    with open(CLUSTER_CONFIG) as f:
        return yaml.safe_load(f) or {}


def get_teacher_for_cluster(cluster_name: str, config: dict) -> str:
    """Get the designated teacher driver for a cluster."""
    teachers = config.get("distillation_teachers", {})
    cluster_teacher = teachers.get(cluster_name, {})
    return cluster_teacher.get("primary", "groq")


def get_fallback_teacher(cluster_name: str, config: dict) -> str:
    """Get fallback teacher driver."""
    teachers = config.get("distillation_teachers", {})
    cluster_teacher = teachers.get(cluster_name, {})
    return cluster_teacher.get("fallback", "gemini")


async def distill_single_agent(
    agent_type: str,
    driver: str,
    fallback_driver: str,
    samples: int,
) -> dict:
    """
    Distill training data for one agent using domain-aware teacher.

    Returns: {agent_type, examples_generated, output_path}
    """
    sys.path.insert(0, str(Path.cwd()))

    config_path = AGENT_CONFIG_DIR / f"{agent_type}.yaml"
    if not config_path.exists():
        return {"agent_type": agent_type, "examples_generated": 0, "error": "config not found"}

    with open(config_path) as f:
        agent_config = yaml.safe_load(f) or {}

    agent_config["_filename"] = agent_type

    # Generate prompts using the agent-aware generator
    from scripts.generate_agent_training_data import generate_prompts_for_agent
    prompts = generate_prompts_for_agent(agent_config, samples)

    # Get system prompt with schema instructions
    system_prompt = agent_config.get("system_prompt", f"You are a {agent_type} agent.")
    schema = agent_config.get("response_schema", {})
    if isinstance(schema, dict) and schema.get("format") == "json":
        fields = schema.get("fields", [])
        fields_str = ", ".join(f'"{f}"' for f in fields)
        system_prompt += f"\n\nRespond with valid JSON only. Required keys: {fields_str}"
    elif isinstance(schema, list):
        fields_str = ", ".join(f'"{f}"' for f in schema)
        system_prompt += f"\n\nRespond with valid JSON only. Required keys: {fields_str}"

    # Call teacher model
    from app.services.llm_service import get_llm_service
    llm = get_llm_service()

    examples = []
    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]

        for attempt_driver in [driver, fallback_driver]:
            try:
                response = await llm.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    driver=attempt_driver,
                )

                if response.content and len(response.content) > 50:
                    examples.append({
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response.content},
                        ],
                        "metadata": {
                            "agent_type": agent_type,
                            "source_driver": attempt_driver,
                            "source_model": response.model,
                            "domain": agent_config.get("domain", "general"),
                            "distilled_at": datetime.now().isoformat(),
                        },
                    })
                    logger.info(f"  [{agent_type}] {i+1}/{samples} via {attempt_driver} ({len(response.content)} chars)")
                    break  # Success, don't try fallback
                else:
                    logger.warning(f"  [{agent_type}] {i+1}/{samples} empty from {attempt_driver}")

            except Exception as e:
                logger.warning(f"  [{agent_type}] {i+1}/{samples} {attempt_driver} failed: {e}")
                continue

    # Apply quality pipeline
    if examples:
        from app.services.intelligence.data_quality import DataQualityPipeline
        quality = DataQualityPipeline()
        result = quality.run(examples, agent_type)
        valid_examples = result["valid_examples"]
    else:
        valid_examples = []

    # Save to per-agent directory
    agent_dir = OUTPUT_DIR / agent_type
    agent_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = agent_dir / f"distilled_{driver}_{timestamp}.jsonl"

    with open(output_path, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    return {
        "agent_type": agent_type,
        "examples_generated": len(valid_examples),
        "examples_raw": len(examples),
        "output_path": str(output_path),
        "teacher": driver,
    }


async def distill_cluster(cluster_name: str, config: dict, samples: int):
    """Distill all agents in a cluster."""
    clusters = config.get("clusters", {})
    cluster = clusters.get(cluster_name)
    if not cluster:
        logger.error(f"Cluster '{cluster_name}' not found")
        return

    agents = cluster.get("agents", [])
    teacher = get_teacher_for_cluster(cluster_name, config)
    fallback = get_fallback_teacher(cluster_name, config)

    logger.info(f"\n{'='*60}")
    logger.info(f"Distilling cluster: {cluster_name} ({len(agents)} agents)")
    logger.info(f"Teacher: {teacher} (fallback: {fallback})")
    logger.info(f"Samples per agent: {samples}")
    logger.info(f"{'='*60}")

    results = []
    for agent_type in agents:
        result = await distill_single_agent(agent_type, teacher, fallback, samples)
        results.append(result)

    # Summary
    total = sum(r["examples_generated"] for r in results)
    logger.info(f"\n{'='*60}")
    logger.info(f"CLUSTER [{cluster_name}] DISTILLATION COMPLETE")
    logger.info(f"Total examples: {total}")
    for r in results:
        status = f"{r['examples_generated']} examples" if r["examples_generated"] > 0 else "SKIPPED"
        logger.info(f"  {r['agent_type']}: {status}")
    logger.info(f"{'='*60}")

    # Check if we have enough to train
    training_defaults = config.get("training_defaults", {})
    min_examples = training_defaults.get("min_examples", 200)
    if total >= min_examples:
        logger.info(f"\nReady for training! Run: python scripts/train_cluster_model.py --cluster {cluster_name}")
    else:
        logger.info(f"\nNeed {min_examples - total} more examples. Run distillation again with more samples.")


async def main():
    parser = argparse.ArgumentParser(description="Run domain-aware distillation cycle")
    parser.add_argument("--cluster", help="Distill a specific cluster")
    parser.add_argument("--agent", help="Distill a single agent")
    parser.add_argument("--all", action="store_true", help="Distill all clusters")
    parser.add_argument("--samples", type=int, default=20, help="Samples per agent")
    args = parser.parse_args()

    config = load_config()

    if args.agent:
        # Find which cluster the agent belongs to
        teacher = "groq"
        fallback = "gemini"
        for cluster_name, cluster in config.get("clusters", {}).items():
            if args.agent in cluster.get("agents", []):
                teacher = get_teacher_for_cluster(cluster_name, config)
                fallback = get_fallback_teacher(cluster_name, config)
                break

        result = await distill_single_agent(args.agent, teacher, fallback, args.samples)
        logger.info(f"\nResult: {json.dumps(result, indent=2)}")

    elif args.cluster:
        await distill_cluster(args.cluster, config, args.samples)

    elif args.all:
        for cluster_name in config.get("clusters", {}):
            await distill_cluster(cluster_name, config, args.samples)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
