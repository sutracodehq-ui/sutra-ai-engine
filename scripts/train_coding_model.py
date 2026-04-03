import asyncio
import logging
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parents[1]))

from app.services.intelligence.live_knowledge import get_live_knowledge_cycle
from app.services.intelligence.lora_trainer import get_lora_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SutraCoder-Train")

async def main():
    """
    Automated cycle to create/update the Sutra-Coder local model.
    1. Distill coding knowledge from frontier cloud models.
    2. Merge into training dataset.
    3. Fine-tune local base model (Qwen2.5-Coder) via Ollama Modelfile injection.
    """
    cycle = get_live_knowledge_cycle()
    trainer = get_lora_trainer()
    
    # Coding topics to learn
    topics = [
        "Advanced async Python patterns and FastAPI best practices 2025",
        "React Hook Form with Zod and TypeScript polymorphic components",
        "Self-hosted AI agent architectures for real-time voice and tool-calling",
        "Unit testing for complex RAG pipelines in Python",
        "SQLAlchemy 2.0 async patterns and PostgreSQL optimization"
    ]
    
    logger.info("--- Phase 1: Distilling Coding Knowledge ---")
    total_examples = 0
    for topic in topics:
        try:
            logger.info(f"Distilling knowledge for: {topic}")
            count = await cycle.run_cycle(domain="coding", query=topic)
            total_examples += count
            logger.info(f"Added {count} examples for {topic}")
        except Exception as e:
            logger.error(f"Failed to distill '{topic}': {e}")
            
    logger.info(f"Distillation complete. Total examples collected: {total_examples}")
    
    logger.info("--- Phase 2: Running Fine-Tuning Pipeline ---")
    # This will pick up all training/data/*.jsonl files and merge them
    result = await trainer.run_pipeline()
    
    if result.get("status") == "success":
        logger.info(f"SUCCESS: New coding model created: {result['new_model']}")
        logger.info(f"Base model used: {result['base_model']}")
    elif result.get("status") == "skipped":
        logger.info(f"SKIPPED: {result.get('reason')}")
    else:
        logger.error(f"FAILURE: Fine-tuning failed: {result.get('error')}")

    await cycle.close()
    await trainer.close()

if __name__ == "__main__":
    asyncio.run(main())
