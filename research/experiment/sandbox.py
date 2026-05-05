"""Experiment sandbox — isolated code execution with metric extraction.

Supports: subprocess, Docker, SSH remote, and Colab Drive modes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sandbox result
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    elapsed_sec: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Base Sandbox
# ---------------------------------------------------------------------------


class ExperimentSandbox:
    """Execute experiment code in isolation and collect metrics."""

    def __init__(
        self,
        python_path: str = "python3",
        timeout_sec: int = 300,
        workdir: Path | None = None,
    ) -> None:
        self.python_path = python_path
        self.timeout_sec = timeout_sec
        self.workdir = workdir or Path(tempfile.mkdtemp(prefix="rc_exp_"))

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int | None = None,
    ) -> SandboxResult:
        """Run an experiment project and collect metrics."""
        timeout = timeout_sec or self.timeout_sec
        entry = project_dir / entry_point

        if not entry.exists():
            return SandboxResult(
                returncode=1,
                stderr=f"Entry point not found: {entry}",
            )

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [self.python_path, str(entry)],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            elapsed = time.monotonic() - start
            metrics = parse_metrics(proc.stdout)
            return SandboxResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                elapsed_sec=elapsed,
                metrics=metrics,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return SandboxResult(
                returncode=-1,
                stderr=f"Experiment timed out after {timeout}s",
                elapsed_sec=elapsed,
                timed_out=True,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return SandboxResult(
                returncode=-1,
                stderr=str(exc),
                elapsed_sec=elapsed,
            )

    def run_code(self, code: str, *, timeout_sec: int | None = None) -> SandboxResult:
        """Run a single code string."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        script = self.workdir / "_run.py"
        script.write_text(code, encoding="utf-8")
        return self.run_project(self.workdir, entry_point="_run.py", timeout_sec=timeout_sec)


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------


def parse_metrics(stdout: str) -> dict[str, Any]:
    """Extract key=value metrics from experiment stdout.

    Recognises patterns:
      metric_name: 0.95
      metric_name = 0.95
      {"metric_name": 0.95, ...}  (JSON lines)
    """
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        # JSON line
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    for k, v in data.items():
                        try:
                            metrics[k] = float(v)
                        except (ValueError, TypeError):
                            metrics[k] = v
                continue
            except json.JSONDecodeError:
                pass

        # key: value or key = value
        m = re.match(r"^([\w\.\-/\s]+?)\s*[:=]\s*([\d\.\-+eE]+)\s*$", line)
        if m:
            name = m.group(1).strip()
            try:
                metrics[name] = float(m.group(2))
            except ValueError:
                pass

    return metrics


# ---------------------------------------------------------------------------
# Paired comparison extraction
# ---------------------------------------------------------------------------


def extract_paired_comparisons(
    run_dir: Path,
    metric_key: str = "primary_metric",
) -> list[dict[str, Any]]:
    """Extract paired metric comparisons across experiment conditions.

    Scans stage-12 and stage-13 output directories for metric JSON files.
    """
    comparisons: list[dict[str, Any]] = []
    for stage_name in ("stage-12", "stage-13"):
        stage_dir = run_dir / stage_name
        if not stage_dir.is_dir():
            continue
        for metrics_file in sorted(stage_dir.glob("**/metrics*.json")):
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and metric_key in data:
                    comparisons.append({
                        "source": str(metrics_file.relative_to(run_dir)),
                        "condition": metrics_file.parent.name,
                        metric_key: data[metric_key],
                        "all_metrics": data,
                    })
            except Exception:
                continue
    return comparisons
