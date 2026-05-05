"""
core/tools/output_manager.py — Save all outputs to organized directory.

Output structure:
  output/<run_id>/
    ├── code/           — generated/improved code files
    ├── experiments/    — experiment results, metrics, graphs
    ├── paper/          — LaTeX paper draft
    ├── knowledge/      — learned insights
    └── summary.json    — run summary
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class OutputManager:
    def __init__(self, base_dir: str = "./output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, task: str) -> "RunOutput":
        run_id = f"run_{int(time.time())}"
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "code").mkdir(exist_ok=True)
        (run_dir / "experiments").mkdir(exist_ok=True)
        (run_dir / "paper").mkdir(exist_ok=True)
        (run_dir / "knowledge").mkdir(exist_ok=True)
        return RunOutput(run_dir, run_id, task)


class RunOutput:
    def __init__(self, run_dir: Path, run_id: str, task: str):
        self.run_dir = run_dir
        self.run_id = run_id
        self.task = task
        self.files_saved: list[str] = []

    def save_code(self, code: str, filename: str = "main.py") -> str:
        path = self.run_dir / "code" / filename
        path.write_text(code, encoding="utf-8")
        self.files_saved.append(str(path))
        logger.info("Saved code: %s", path)
        return str(path)

    def save_experiment_results(self, results: str, filename: str = "results.md") -> str:
        path = self.run_dir / "experiments" / filename
        path.write_text(results, encoding="utf-8")
        self.files_saved.append(str(path))
        return str(path)

    def save_experiment_script(self, script: str, filename: str = "train.py") -> str:
        path = self.run_dir / "experiments" / filename
        path.write_text(script, encoding="utf-8")
        self.files_saved.append(str(path))
        return str(path)

    def save_paper(self, latex: str, filename: str = "paper.tex") -> str:
        path = self.run_dir / "paper" / filename
        path.write_text(latex, encoding="utf-8")
        self.files_saved.append(str(path))
        return str(path)

    def save_knowledge(self, knowledge: str, filename: str = "insights.md") -> str:
        path = self.run_dir / "knowledge" / filename
        path.write_text(knowledge, encoding="utf-8")
        self.files_saved.append(str(path))
        return str(path)

    def save_summary(self, results: list[dict], passed: bool, elapsed: float) -> str:
        def _match_category(filepath: str, category: str) -> bool:
            """OS-agnostic check for category subdirectory in path."""
            normalized = filepath.replace("\\", "/")
            return f"/{category}/" in normalized

        summary = {
            "run_id": self.run_id,
            "task": self.task,
            "passed": passed,
            "elapsed_sec": elapsed,
            "steps": len(results),
            "files_saved": self.files_saved,
            "outputs": {
                "code": [f for f in self.files_saved if _match_category(f, "code")],
                "experiments": [f for f in self.files_saved if _match_category(f, "experiments")],
                "paper": [f for f in self.files_saved if _match_category(f, "paper")],
                "knowledge": [f for f in self.files_saved if _match_category(f, "knowledge")],
            },
        }
        path = self.run_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2))
        return str(path)
