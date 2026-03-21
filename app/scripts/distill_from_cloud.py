"""
Distillation Script — Generate training data from premium cloud models.

Intelligence Upgrade 5: Uses agent YAML configs to generate synthetic
prompts, sends them to premium models (Groq/Gemini/Anthropic), and
saves the responses as JSONL training data for LoRA fine-tuning.

Usage:
    python -m app.scripts.distill_from_cloud --agents seo copywriter quiz_generator
    python -m app.scripts.distill_from_cloud --all --samples 20
"""

import argparse
import asyncio
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AGENT_CONFIG_DIR = Path("agent_config")
OUTPUT_DIR = Path("training/data")

# Synthetic prompt templates per domain
PROMPT_TEMPLATES: dict[str, list[str]] = {
    "edtech": [
        "Generate a quiz from these notes on {topic}",
        "Create flashcards for the chapter on {topic}",
        "Summarize the key points of {topic} for revision",
        "Plan a 45-minute lecture on {topic} for undergraduate students",
        "Extract the most important concepts from this text about {topic}",
    ],
    "SEO, meta tags, keywords, content structure, technical SEO, and page performance": [
        "Optimize SEO for a blog post about {topic}",
        "Generate meta tags and keywords for a landing page on {topic}",
        "Create a content outline optimized for search ranking on {topic}",
        "Audit this page content for SEO improvements: {topic}",
        "Generate schema markup for a {topic} page",
    ],
    "default": [
        "Create professional content about {topic}",
        "Analyze and provide recommendations for {topic}",
        "Generate a detailed report on {topic}",
        "Write a comprehensive strategy for {topic}",
        "Provide expert advice on {topic}",
    ],
}

SAMPLE_TOPICS = [
    "photosynthesis", "machine learning basics", "world war 2", "digital marketing",
    "python programming", "climate change", "human anatomy", "blockchain technology",
    "artificial intelligence", "quantum computing", "data science", "web development",
    "social media marketing", "content strategy", "email marketing campaigns",
    "brand positioning", "competitive analysis for startups", "customer retention",
    "e-commerce optimization", "influencer marketing strategy",
]


def load_agent_configs(agent_types: list[str] | None = None) -> list[dict]:
    """Load agent YAML configs."""
    configs = []
    for yaml_file in AGENT_CONFIG_DIR.glob("*.yaml"):
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        if config:
            config["_filename"] = yaml_file.stem
            if agent_types is None or yaml_file.stem in agent_types:
                configs.append(config)
    return configs


def generate_prompts(config: dict, n_samples: int = 10) -> list[str]:
    """Generate synthetic prompts for an agent based on its domain."""
    domain = config.get("domain", "default")
    templates = PROMPT_TEMPLATES.get(domain, PROMPT_TEMPLATES["default"])

    prompts = []
    for i in range(n_samples):
        template = templates[i % len(templates)]
        topic = SAMPLE_TOPICS[i % len(SAMPLE_TOPICS)]
        prompts.append(template.format(topic=topic))

    return prompts


async def distill_agent(config: dict, n_samples: int = 10, driver: str = "groq") -> list[dict]:
    """Generate training data for one agent using a cloud model."""
    from app.services.llm_service import get_llm_service

    agent_type = config.get("_filename", "unknown")
    system_prompt = config.get("system_prompt", f"You are a {agent_type} agent.")

    # Inject response schema instructions
    schema = config.get("response_schema", {})
    if isinstance(schema, dict) and schema.get("format") == "json":
        fields = schema.get("fields", [])
        fields_str = ", ".join(f'"{f}"' for f in fields)
        system_prompt += f"\n\nRespond with valid JSON only. Required keys: {fields_str}"
    elif isinstance(schema, list):
        fields_str = ", ".join(f'"{f}"' for f in schema)
        system_prompt += f"\n\nRespond with valid JSON only. Required keys: {fields_str}"

    prompts = generate_prompts(config, n_samples)
    llm = get_llm_service()
    examples = []

    for i, prompt in enumerate(prompts):
        try:
            response = await llm.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                driver=driver,
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
                        "source_driver": driver,
                        "source_model": response.model,
                    },
                })
                logger.info(f"  [{agent_type}] {i+1}/{n_samples} ✓ ({len(response.content)} chars)")
            else:
                logger.warning(f"  [{agent_type}] {i+1}/{n_samples} ✗ (empty/short)")

        except Exception as e:
            logger.error(f"  [{agent_type}] {i+1}/{n_samples} ✗ Error: {e}")

    return examples


async def main(agent_types: list[str] | None, n_samples: int, driver: str):
    """Main distillation pipeline."""
    configs = load_agent_configs(agent_types)
    logger.info(f"Distilling {len(configs)} agents, {n_samples} samples each, via {driver}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"distilled_{driver}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    all_examples = []
    for config in configs:
        agent_type = config.get("_filename", "unknown")
        logger.info(f"\n{'='*50}")
        logger.info(f"Distilling: {agent_type}")
        logger.info(f"{'='*50}")

        examples = await distill_agent(config, n_samples, driver)
        all_examples.extend(examples)

    # Write JSONL
    with open(output_file, "w") as f:
        for ex in all_examples:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    # Summary
    by_agent: dict[str, int] = {}
    for ex in all_examples:
        a = ex["metadata"]["agent_type"]
        by_agent[a] = by_agent.get(a, 0) + 1

    logger.info(f"\n{'='*50}")
    logger.info(f"DISTILLATION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"By agent: {json.dumps(by_agent, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill training data from cloud models")
    parser.add_argument("--agents", nargs="+", help="Agent types to distill", default=None)
    parser.add_argument("--all", action="store_true", help="Distill all agents")
    parser.add_argument("--samples", type=int, default=10, help="Samples per agent")
    parser.add_argument("--driver", default="groq", choices=["groq", "gemini", "anthropic"])
    args = parser.parse_args()

    agent_types = None if args.all else args.agents
    asyncio.run(main(agent_types, args.samples, args.driver))
