#!/usr/bin/env python3
"""
Agent Evaluator — Automated benchmarking for AI agents.

This script runs an agent against a set of questions and evaluates the responses
based on schema compliance, key information retrieval, and stylistic consistency.

Usage:
    python scripts/evaluate_agent.py --agent-type quiz_generator --samples 5
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from colorama import Fore, Style, init

from app.services.agents.hub import get_agent_hub
from app.services.intelligence.agent_evaluator import get_agent_evaluator

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Evaluate an AI agent's performance.")
    parser.add_argument("--agent-type", type=str, required=True, help="Type of agent to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Specific model to test (Ollama name)")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--gold", type=str, default=None, help="Path to gold dataset (JSONL)")
    parser.add_argument("--output", type=str, default=None, help="Output results path")
    args = parser.parse_args()

    hub = get_agent_hub()
    evaluator = get_agent_evaluator()
    
    agent = hub.get(args.agent_type)
    if not agent:
        print(f"{Fore.RED}Error: Agent '{args.agent_type}' not found.{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}--- Evaluating Agent: {args.agent_type} ---{Style.RESET_ALL}")
    
    # ─── 1. Load Gold Dataset or Generate Prompts ──────────
    prompts = []
    if args.gold:
        gold_path = Path(args.gold)
        if gold_path.exists():
            with open(gold_path) as f:
                for line in f:
                    if line.strip():
                        prompts.append(json.loads(line))
        else:
            print(f"{Fore.YELLOW}Warning: Gold path {args.gold} not found. Generating dummy prompts.{Style.RESET_ALL}")
    
    if not prompts:
        # Fallback: simple generic prompts based on agent type
        prompts = [{"prompt": f"Generate a high-quality response for {args.agent_type}", "expected_fields": []} for _ in range(args.samples)]

    # ─── 2. Run Evaluation ────────────────────────────────
    results = []
    passed = 0
    total_score = 0.0

    print(f"Running {len(prompts)} tests...")

    for i, test in enumerate(prompts):
        prompt_text = test.get("prompt", "")
        expected_fields = test.get("expected_fields", [])
        
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt_text[:60]}...")
        
        try:
            # Execute agent (using model override if provided)
            options = {}
            if args.model:
                options["model"] = args.model
            
            # Note: We call agent.execute directly to bypass HybridRouter if we want to test a specific model
            response = await agent.execute(prompt_text, **options)
            
            # Evaluate result
            eval_result = evaluator.evaluate(
                agent_type=args.agent_type,
                response=response.content,
                expected_fields=expected_fields
            )
            
            score = eval_result.get("total", 0)
            status = f"{Fore.GREEN}PASS" if eval_result.get("passed") else f"{Fore.RED}FAIL"
            
            print(f"  Status: {status} {Style.RESET_ALL}(Score: {score}/10)")
            
            results.append({
                "test_id": i + 1,
                "prompt": prompt_text,
                "response": response.content,
                "eval": eval_result,
                "latency_ms": response.metadata.get("latency_ms", 0) if response.metadata else 0,
                "model": response.model
            })
            
            total_score += score
            if eval_result.get("passed"):
                passed += 1

        except Exception as e:
            print(f"  {Fore.RED}Error: {e}{Style.RESET_ALL}")
            results.append({
                "test_id": i + 1,
                "prompt": prompt_text,
                "error": str(e)
            })

    # ─── 3. Final Report ──────────────────────────────────
    avg_score = total_score / len(prompts) if prompts else 0
    pass_rate = (passed / len(prompts) * 100) if prompts else 0

    print(f"\n{Fore.GREEN}{Style.BRIGHT}=== Evaluation Summary ==={Style.RESET_ALL}")
    print(f"Agent:      {args.agent_type}")
    print(f"Model:      {args.model or 'Default'}")
    print(f"Pass Rate:  {pass_rate:.1f}% ({passed}/{len(prompts)})")
    print(f"Avg Score:  {avg_score:.1f}/10")

    # ─── 4. Save Results ──────────────────────────────────
    res_dir = Path("training/benchmarks/results")
    res_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else res_dir / f"eval_{args.agent_type}_{timestamp}.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "agent_type": args.agent_type,
        "model": args.model,
        "summary": {
            "total_tests": len(prompts),
            "passed": passed,
            "pass_rate": pass_rate,
            "avg_score": avg_score
        },
        "details": results
    }
    
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{Fore.CYAN}Full report saved to: {out_path}{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())
