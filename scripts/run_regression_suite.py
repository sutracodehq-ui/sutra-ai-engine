#!/usr/bin/env python3
"""
Regression Suite — Comprehensive AI Engine regression testing.

Runs all core agents against their gold-standard datasets and generates
a unified performance report.

Software Factory Principle: Quality-gated progression.

Usage:
    python scripts/run_regression_suite.py --model llama3.2:1b
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from colorama import Fore, Style, init

init(autoreset=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

async def run_eval(agent_type: str, model: str | None = None) -> dict:
    """Run evaluate_agent.py script for a specific agent and return the report."""
    gold_path = Path(f"training/benchmarks/gold/{agent_type}.jsonl")
    if not gold_path.exists():
        return {"status": "skipped", "reason": f"Gold dataset missing for {agent_type}"}

    cmd = [sys.executable, "scripts/evaluate_agent.py", "--agent-type", agent_type]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["--gold", str(gold_path), "--output", "/tmp/current_eval.json"])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            return {"status": "error", "reason": stderr.decode()}

        with open("/tmp/current_eval.json") as f:
            return json.load(f)

    except Exception as e:
        return {"status": "error", "reason": str(e)}

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run regression suite for AI agents.")
    parser.add_argument("--model", type=str, default=None, help="Specific model to test")
    parser.add_argument("--all", action="store_true", help="Run all agents found in hub")
    args = parser.parse_args()

    gold_dir = Path("training/benchmarks/gold")
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    # ─── 1. Identify Agents to Test ──────────────────────────
    agents_to_test = []
    for gold_file in gold_dir.glob("*.jsonl"):
        agents_to_test.append(gold_file.stem)
    
    if not agents_to_test:
        print(f"{Fore.RED}Error: No gold datasets found in {gold_dir}.{Style.RESET_ALL}")
        print(f"Please add some *.jsonl files (e.g., quiz_generator.jsonl) to the directory.")
        return

    print(f"{Fore.CYAN}=== Starting AI Regression Suite ==={Style.RESET_ALL}")
    print(f"Model: {args.model or 'Default'}")
    print(f"Found {len(agents_to_test)} agents with gold datasets.\n")

    # ─── 2. Run Evaluations ───────────────────────────────
    reports = {}
    overall_passed = 0
    overall_total = 0

    for agent in agents_to_test:
        print(f"Testing {Fore.YELLOW}{agent}{Style.RESET_ALL}...", end="", flush=True)
        report = await run_eval(agent, args.model)
        
        if report.get("summary"):
            s = report["summary"]
            pass_rate = s["pass_rate"]
            status = f"{Fore.GREEN}PASS" if pass_rate >= 80 else f"{Fore.RED}FAIL"
            print(f" {status}{Style.RESET_ALL} ({pass_rate:.1f}%)")
            
            reports[agent] = report
            overall_passed += s["passed"]
            overall_total += s["total_tests"]
        else:
            print(f" {Fore.RED}ERROR{Style.RESET_ALL} - {report.get('reason')}")

    # ─── 3. Final Aggregated Report ────────────────────────
    total_pass_rate = (overall_passed / overall_total * 100) if overall_total > 0 else 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "summary": {
            "total_agents": len(reports),
            "total_tests": overall_total,
            "overall_passed": overall_passed,
            "overall_pass_rate": total_pass_rate
        },
        "agents": reports
    }

    # Save Aggregate Results
    res_dir = Path("training/benchmarks/results")
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / f"regression_{timestamp}.json"
    
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2)

    # ─── 4. Generate Markdown Summary ─────────────────────
    md_path = res_dir / f"regression_summary_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Regression Report - {timestamp}\n\n")
        f.write(f"- **Overall Pass Rate**: {total_pass_rate:.1f}%\n")
        f.write(f"- **Model**: {args.model or 'Default'}\n")
        f.write(f"- **Agents Tested**: {len(reports)}\n\n")
        f.write("| Agent | Pass Rate | Status |\n")
        f.write("|-------|-----------|--------|\n")
        for agent, report in reports.items():
            rate = report['summary']['pass_rate']
            status = "✅ PASS" if rate >= 80 else "❌ FAIL"
            f.write(f"| {agent} | {rate:.1f}% | {status} |\n")

    print(f"\n{Fore.GREEN}{Style.BRIGHT}=== Global Summary ==={Style.RESET_ALL}")
    print(f"Total Agents:  {len(reports)}")
    print(f"Pass Rate:     {total_pass_rate:.1f}%")
    print(f"Report:        {out_path}")
    print(f"Dashboard:     {md_path}")

if __name__ == "__main__":
    asyncio.run(main())
