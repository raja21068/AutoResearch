"""
run_pipeline.py — End-to-end research pipeline runner (no Docker required).

Usage:
    python run_pipeline.py --topic "Your research topic here"
    python run_pipeline.py --topic "Efficient attention mechanisms" --mode full
    python run_pipeline.py --topic "..." --mode paper_only

Modes:
    full       — Ideation → Code → Test → Experiment → Paper (default)
    code_only  — Ideation → Code → Test only
    paper_only — Uses existing experiment data to write paper

Requirements:
    - DeepSeek API key in .env (DEEPSEEK_API_KEY)
    - Python packages: pip install -r requirements.txt
    - No Docker needed — uses subprocess sandbox

Windows/macOS/Linux compatible.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Windows asyncio fix ──────────────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
load_dotenv()

from orchestrator import Orchestrator
from llm import preflight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


async def run_full_pipeline(topic: str, mode: str = "full") -> dict:
    """Run the complete AutoResearch pipeline."""
    print(f"\n{'='*70}")
    print(f"  AutoResearch Pipeline — {mode.upper()} mode")
    print(f"  Topic: {topic}")
    print(f"{'='*70}\n")

    # ── Preflight: verify API connectivity ──
    print("[1/7] Preflight check...")
    ok, msg = await preflight()
    if not ok:
        print(f"  FAIL: {msg}")
        print("  Check your .env file — DEEPSEEK_API_KEY must be set.")
        return {"error": msg}
    print(f"  OK: {msg}")

    # ── Initialize orchestrator ──
    orch = Orchestrator()

    # ── Build task description based on mode ──
    if mode == "full":
        task = (
            f"Research topic: {topic}\n\n"
            "Execute the FULL research pipeline:\n"
            "1. [researcher] Search literature on ArXiv for this topic. "
            "Find 10-15 relevant recent papers.\n"
            "2. [coder] Write a lightweight, self-contained PyTorch implementation "
            "that demonstrates the key concept. Use synthetic data (torch.randn) — "
            "do NOT download any datasets or models from the internet. "
            "Code must run in under 60 seconds.\n"
            "3. [tester] Write and run tests for the implementation.\n"
            "4. [experiment] Design and analyze experiments with the code.\n"
            "5. [paper_writer] Write a complete publication-ready research paper "
            "with all standard sections.\n"
            "6. [critic] Review the final output."
        )
    elif mode == "code_only":
        task = (
            f"Research topic: {topic}\n\n"
            "Write a clean, self-contained PyTorch implementation that demonstrates "
            "the key concept. Use synthetic data — no downloads. Must run in 60s.\n"
            "Then test it thoroughly."
        )
    elif mode == "paper_only":
        task = (
            f"Research topic: {topic}\n\n"
            "Execute a paper-writing pipeline:\n"
            "1. [researcher] Search literature on this topic.\n"
            "2. [paper_writer] Write a complete research paper with: "
            "Abstract, Introduction, Related Work, Methodology, "
            "Experiments, Results, Discussion, and Conclusion.\n"
            "3. [critic] Review the paper."
        )
    else:
        task = f"Research topic: {topic}\n\nExecute research and produce results."

    # ── Run pipeline with live progress ──
    print(f"\n[2/7] Running pipeline...\n")
    step_count = 0

    async def progress_callback(event: str, data: str):
        nonlocal step_count
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            parsed = {"raw": str(data)[:200]}

        if event == "agent":
            agent = parsed.get("agent", "?")
            status = parsed.get("status", "?")
            step = parsed.get("step", "")[:80]
            if status == "running":
                step_count += 1
                print(f"  [{step_count}] {agent}: {step}")
            elif status == "done":
                print(f"      -> done")
        elif event == "test":
            passed = parsed.get("passed", False)
            symbol = "PASS" if passed else "FAIL"
            print(f"      -> Tests: {symbol}")
        elif event == "review":
            review = parsed.get("review", "")[:200]
            passed = parsed.get("passed", False)
            print(f"\n[6/7] Final Review: {'PASS' if passed else 'FAIL'}")
            print(f"      {review}")
        elif event == "complete":
            elapsed = parsed.get("elapsed_sec", 0)
            print(f"\n[7/7] Complete in {elapsed:.1f}s")

    t0 = time.time()
    result = await orch.run(task, callback=progress_callback)
    elapsed = time.time() - t0

    # ── Report ──
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Passed:    {result.get('passed', False)}")
    print(f"  Steps:     {len(result.get('results', []))}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"  Output:    {result.get('output_dir', 'N/A')}")

    out_dir = result.get("output_dir", "")
    if out_dir:
        print(f"\n  Files saved in: {out_dir}/")
        for category in ["code", "experiments", "paper", "knowledge"]:
            cat_dir = Path(out_dir) / category
            if cat_dir.exists():
                files = list(cat_dir.iterdir())
                if files:
                    print(f"    {category}/")
                    for f in files:
                        print(f"      - {f.name} ({f.stat().st_size} bytes)")

    # ── Shutdown ──
    orch.shutdown()
    print(f"\n{'='*70}")
    print(f"  Pipeline complete.")
    print(f"{'='*70}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch Pipeline Runner (no Docker required)")
    parser.add_argument("--topic", required=True,
                        help="Research topic to investigate")
    parser.add_argument("--mode", default="full",
                        choices=["full", "code_only", "paper_only"],
                        help="Pipeline mode (default: full)")
    args = parser.parse_args()

    result = asyncio.run(run_full_pipeline(args.topic, args.mode))
    sys.exit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
