import asyncio
import logging
import sys
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.intelligence.lora_trainer import get_lora_trainer
from app.services.intelligence.live_knowledge import KnowledgeSeeker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Mock Search Data ---
MOCK_SEARCH_RESULTS = [
    {
        "url": "https://deepseek.com/blog/v3-158bit",
        "title": "DeepSeek-V3: High-Performance 1.58-bit Quantization",
        "content": "DeepSeek-V3 introduces a revolutionary 1.58-bit quantization technique that maintains 98% of full-precision performance while reducing memory usage by 4x. This is achieved through a multi-stage weight-only quantization and a novel outlier handling mechanism in LUT kernels."
    },
    {
        "url": "https://arxiv.org/abs/2412.0000",
        "title": "Scalable 1-bit Training for Large Language Models",
        "content": "The BitNet b1.58 architecture proves that LLMs can be trained with pure ternary weights {-1, 0, 1}. This paper demonstrates that 1.58-bit models scale significantly better than 8-bit or 4-bit models for the same compute budget."
    }
]

async def test_live_cycle():
    trainer = get_lora_trainer()
    
    # Test query: Something very recent/specific
    domain = "technology"
    query = "DeepSeek-V3 architecture and 1.58-bit quantization breakthroughs 2024 2025"
    
    print(f"\n🚀 Phase 1: Running Self-Improvement Cycle (MOCK SEARCH) for '{query}'...")
    
    # Patch KnowledgeSeeker.search to bypass the 401 and demonstrate the REST of the pipeline
    with patch.object(KnowledgeSeeker, 'search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = MOCK_SEARCH_RESULTS
        
        result = await trainer.run_self_improvement_cycle(domain, query)
    
    print("\n--- Pipeline Result ---")
    print(json.dumps(result, indent=2, default=str))
    
    if result.get("status") == "success":
        print(f"\n✅ Success! Pipeline reached final deployment.")
        print(f"📊 Examples synthesized via Gemini: {result.get('new_knowledge_count')}")
        print(f"🛠️ New Model Variant: {result.get('new_model')}")
        if "benchmark" in result:
            print(f"📈 Pre-deployment Quality (via EvolutionEngine): {result['benchmark']['quality']}/10")
    else:
        print(f"\n⚠️ Cycle skipped/failed: {result.get('reason', 'Unknown error')}")

if __name__ == "__main__":
    try:
        asyncio.run(test_live_cycle())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
