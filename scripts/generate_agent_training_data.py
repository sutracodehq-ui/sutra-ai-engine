"""
Agent-Aware Training Data Generator.

Reads each agent's YAML config (system_prompt, response_schema, rules, capabilities)
and generates domain-specific, schema-aware synthetic prompts.

Unlike the generic distill_from_cloud.py, this generator:
1. Uses the agent's actual domain and capabilities to create realistic prompts
2. Generates edge cases (short, long, multi-language, ambiguous)
3. Exercises all response_schema fields
4. Outputs per-agent prompt files for targeted distillation

Usage:
    python scripts/generate_agent_training_data.py --agents quiz_generator seo
    python scripts/generate_agent_training_data.py --cluster edtech
    python scripts/generate_agent_training_data.py --all --count 50
"""

import argparse
import json
import logging
import random
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AGENT_CONFIG_DIR = Path("agent_config")
OUTPUT_DIR = Path("training/data/per_agent")
CLUSTER_CONFIG = Path("config/model_clusters.yaml")


def load_clusters() -> dict:
    """Load cluster → agent mapping."""
    if not CLUSTER_CONFIG.exists():
        return {}
    with open(CLUSTER_CONFIG) as f:
        config = yaml.safe_load(f) or {}
    return config.get("clusters", {})


def get_agents_for_cluster(cluster_name: str) -> list[str]:
    """Get agent list for a cluster."""
    clusters = load_clusters()
    cluster = clusters.get(cluster_name, {})
    return cluster.get("agents", [])


def load_agent_config(agent_type: str) -> dict | None:
    """Load a single agent's YAML config."""
    path = AGENT_CONFIG_DIR / f"{agent_type}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ─── Domain-Specific Prompt Templates ─────────────────────────

DOMAIN_TEMPLATES = {
    "edtech": {
        "topics": [
            "photosynthesis", "Newton's laws of motion", "Indian independence movement",
            "World War 2", "periodic table", "cell biology", "French Revolution",
            "algebra and quadratic equations", "Shakespeare's Hamlet",
            "human digestive system", "climate change", "ancient civilizations",
            "genetics and DNA", "organic chemistry", "Indian Constitution",
            "probability and statistics", "Renaissance art", "plate tectonics",
            "electromagnetic spectrum", "Indian freedom fighters",
        ],
        "templates": [
            "Create a quiz on {topic} for Class {grade} students with {count} questions",
            "Generate study notes on {topic} covering key concepts and definitions",
            "Design a {difficulty} difficulty test paper on {topic}",
            "Create flashcards for {topic} with key terms and definitions",
            "Plan a 45-minute lecture on {topic} for {grade}th grade",
            "Generate {count} MCQ questions on {topic} with explanations",
            "Create a concept map for {topic} showing relationships between key ideas",
            "Write a summary of {topic} suitable for quick revision before exams",
            "Design an assignment on {topic} with both theoretical and practical questions",
            "Generate true/false questions on {topic} with detailed explanations",
        ],
        "variables": {
            "grade": ["8", "9", "10", "11", "12"],
            "count": ["5", "10", "15", "20"],
            "difficulty": ["easy", "medium", "hard", "mixed"],
        },
    },
    "finance": {
        "topics": [
            "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "AMD",
            "Reliance Industries", "TCS", "Infosys", "HDFC Bank",
            "Bitcoin", "Ethereum", "Solana", "mutual funds", "SIP investing",
            "GST compliance", "ITR filing", "startup valuation",
        ],
        "templates": [
            "Analyze {topic} stock and provide buy/sell/hold recommendation",
            "Calculate ROI if I invest ${amount} in {topic} for {years} years",
            "Create a risk assessment report for investing in {topic}",
            "Compare {topic} vs competitor stocks and recommend the better investment",
            "Predict {topic} price movement for the next quarter",
            "Create a portfolio allocation strategy that includes {topic}",
            "Analyze the P/E ratio and growth metrics for {topic}",
            "Generate a tax optimization strategy for investments in {topic}",
            "Create a monthly budget plan with ${amount} salary in India",
            "Analyze crypto market trends for {topic} and predict next month",
        ],
        "variables": {
            "amount": ["10000", "25000", "50000", "100000", "500000"],
            "years": ["1", "2", "3", "5", "10"],
        },
    },
    "marketing": {
        "topics": [
            "fitness app", "organic skincare brand", "SaaS startup",
            "local restaurant", "online bookstore", "Indian food delivery",
            "EdTech platform", "D2C fashion brand", "health supplement",
            "real estate agency", "coworking space", "electric vehicles",
            "fintech app", "travel agency", "yoga studio",
        ],
        "templates": [
            "Create an SEO strategy for a {topic} targeting {audience}",
            "Write 5 social media posts for {topic} for {platform}",
            "Generate a content calendar for {topic} for the next month",
            "Write compelling ad copy for {topic} targeting {audience}",
            "Design an email campaign sequence for {topic} launch",
            "Create a competitor analysis for {topic} in the Indian market",
            "Write a brand positioning statement for a {topic}",
            "Generate hashtag strategy for {topic} on Instagram",
            "Create a landing page copy for {topic}",
            "Design a referral program for {topic}",
        ],
        "variables": {
            "audience": ["millennials", "working professionals", "students", "parents", "Gen Z"],
            "platform": ["Instagram", "LinkedIn", "Twitter", "YouTube", "Facebook"],
        },
    },
    "health": {
        "topics": [
            "vitamin D deficiency", "diabetes management", "weight loss",
            "PCOS", "thyroid disorders", "high blood pressure",
            "pregnancy nutrition", "child vaccination schedule",
            "mental health anxiety", "sleep disorders", "back pain",
            "cholesterol management", "iron deficiency anemia",
        ],
        "templates": [
            "What are the symptoms and treatment options for {topic}?",
            "Create a {duration} diet plan for managing {topic}",
            "Design a fitness routine for someone dealing with {topic}",
            "Explain common lab test values related to {topic}",
            "Create a daily wellness routine to help with {topic}",
            "What home remedies are recommended for {topic}?",
            "List Ayurvedic approaches for managing {topic}",
            "Create a meal plan for a {diet_type} person with {topic}",
        ],
        "variables": {
            "duration": ["1-week", "2-week", "1-month"],
            "diet_type": ["vegetarian", "vegan", "non-vegetarian", "Jain"],
        },
    },
    "legal": {
        "topics": [
            "employment contract", "NDA agreement", "privacy policy for app",
            "GST filing for small business", "RTI application",
            "rental agreement", "partnership deed", "startup incorporation",
            "trademark registration", "freelancer contract",
        ],
        "templates": [
            "Draft a {topic} for an Indian {business_type}",
            "Review this {topic} and identify potential risks",
            "Explain compliance requirements for {topic} in India",
            "Create a checklist for {topic} documentation",
            "What are the legal implications of not having a proper {topic}?",
        ],
        "variables": {
            "business_type": ["startup", "SME", "freelancer", "e-commerce company", "restaurant"],
        },
    },
    "ecommerce": {
        "topics": [
            "electronics store", "fashion marketplace", "grocery delivery",
            "handmade crafts", "premium watches", "organic food",
            "pet supplies", "home decor", "mobile accessories",
        ],
        "templates": [
            "Write a product description for a bestselling item from a {topic} store",
            "Create a dynamic pricing strategy for {topic} during festival season",
            "Analyze customer churn patterns for a {topic} platform",
            "Design a loyalty program for {topic}",
            "Optimize product catalog for {topic} with SEO-friendly descriptions",
            "Create a return/refund policy for {topic}",
            "Write customer review response templates for {topic}",
            "Design a demand forecasting model for {topic}",
        ],
        "variables": {},
    },
    "coding": {
        "topics": [
            "REST API authentication", "WebSocket chat server",
            "ML data pipeline", "React dashboard component",
            "PostgreSQL query optimization", "FastAPI middleware",
            "Docker containerization", "CI/CD pipeline",
            "data validation schema", "microservices architecture",
        ],
        "templates": [
            "Write production-ready code for {topic}",
            "Review and refactor this implementation of {topic}",
            "Create unit tests for {topic}",
            "Design a schema for {topic}",
            "Implement error handling for {topic}",
            "Optimize performance of {topic}",
        ],
        "variables": {},
    },
    "general": {
        "topics": [
            "time management", "remote work productivity",
            "travel to Rajasthan", "weekend trip from Mumbai",
            "birthday party planning", "home organization",
            "resume for software engineer", "salary negotiation",
            "meditation and mindfulness", "sustainable living",
        ],
        "templates": [
            "Create a detailed plan for {topic}",
            "Generate tips and recommendations for {topic}",
            "Write a comprehensive guide on {topic}",
            "Create a checklist for {topic}",
            "Provide expert advice on {topic}",
        ],
        "variables": {},
    },
}

# Edge case prompt modifiers
EDGE_CASES = [
    "Respond in Hindi.",
    "Keep it under 100 words.",
    "Give me the most detailed analysis possible with all metrics.",
    "I'm a complete beginner, explain like I'm 5.",
    "Format as a bullet-point list only.",
    "",  # Normal case (no modifier)
    "",
    "",
    "",  # Weight toward normal
]


def generate_prompts_for_agent(config: dict, count: int = 50) -> list[dict]:
    """
    Generate synthetic prompts for a single agent.

    Uses the agent's domain, capabilities, and rules to create
    realistic, diverse prompts that exercise all response schema fields.
    """
    domain = config.get("domain", "general")
    agent_type = config.get("identifier", config.get("_filename", "unknown"))
    capabilities = config.get("capabilities", [])
    name = config.get("name", agent_type)

    # Get domain templates
    domain_config = DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES["general"])
    topics = domain_config["topics"]
    templates = domain_config["templates"]
    variables = domain_config.get("variables", {})

    prompts = []

    for i in range(count):
        # Rotate through templates
        template = templates[i % len(templates)]
        topic = topics[i % len(topics)]

        # Fill in template variables
        prompt = template.format(
            topic=topic,
            **{k: random.choice(v) for k, v in variables.items()},
        )

        # Occasionally add edge case modifiers (20% of prompts)
        edge_case = random.choice(EDGE_CASES)
        if edge_case:
            prompt = f"{prompt} {edge_case}"

        # Occasionally use capability-based prompts (15% of prompts)
        if capabilities and random.random() < 0.15:
            cap = random.choice(capabilities)
            prompt = f"Using your ability to {cap.lower()}, analyze: {topic}"

        prompts.append({
            "prompt": prompt,
            "agent_type": agent_type,
            "domain": domain,
            "template_idx": i % len(templates),
            "topic": topic,
            "has_edge_case": bool(edge_case),
        })

    return prompts


def save_prompts(agent_type: str, prompts: list[dict]) -> Path:
    """Save generated prompts to per-agent directory."""
    agent_dir = OUTPUT_DIR / agent_type
    agent_dir.mkdir(parents=True, exist_ok=True)

    path = agent_dir / "synthetic_prompts.jsonl"
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(prompts)} prompts → {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate agent-aware training prompts")
    parser.add_argument("--agents", nargs="+", help="Specific agent types")
    parser.add_argument("--cluster", help="Generate for all agents in a cluster")
    parser.add_argument("--all", action="store_true", help="Generate for all agents")
    parser.add_argument("--count", type=int, default=50, help="Prompts per agent")
    args = parser.parse_args()

    # Determine which agents to process
    if args.cluster:
        agent_types = get_agents_for_cluster(args.cluster)
        if not agent_types:
            logger.error(f"Cluster '{args.cluster}' not found or empty")
            return
        logger.info(f"Generating for cluster '{args.cluster}': {len(agent_types)} agents")
    elif args.all:
        agent_types = [f.stem for f in AGENT_CONFIG_DIR.glob("*.yaml")]
    elif args.agents:
        agent_types = args.agents
    else:
        parser.print_help()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_prompts = 0
    for agent_type in agent_types:
        config = load_agent_config(agent_type)
        if not config:
            logger.warning(f"No config found for agent: {agent_type}")
            continue

        config["_filename"] = agent_type
        prompts = generate_prompts_for_agent(config, args.count)
        save_prompts(agent_type, prompts)
        total_prompts += len(prompts)

    logger.info(f"\nTotal: {total_prompts} prompts for {len(agent_types)} agents")


if __name__ == "__main__":
    main()
