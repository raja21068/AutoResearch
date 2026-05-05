"""ExperimentRunner — iterative run→evaluate→refine loop."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research.experiment.sandbox import ExperimentSandbox, SandboxResult

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    iteration: int
    result: SandboxResult
    metrics: dict[str, Any] = field(default_factory=dict)
    improved: bool = False


class ExperimentRunner:
    """Run experiments iteratively, tracking metrics across iterations."""

    def __init__(
        self,
        sandbox: ExperimentSandbox,
        *,
        max_iterations: int = 10,
        metric_key: str = "primary_metric",
        metric_direction: str = "minimize",
        time_budget_sec: int = 300,
    ) -> None:
        self.sandbox = sandbox
        self.max_iterations = max_iterations
        self.metric_key = metric_key
        self.metric_direction = metric_direction
        self.time_budget_sec = time_budget_sec
        self.history: list[RunRecord] = []
        self.best_metric: float | None = None
        self.best_iteration: int = -1

    def run_once(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        iteration: int = 0,
    ) -> RunRecord:
        """Execute a single run and record results."""
        result = self.sandbox.run_project(
            project_dir, entry_point=entry_point,
            timeout_sec=self.time_budget_sec,
        )
        metrics = result.metrics
        improved = self._check_improvement(metrics)

        record = RunRecord(
            iteration=iteration,
            result=result,
            metrics=metrics,
            improved=improved,
        )
        self.history.append(record)

        if improved:
            self.best_iteration = iteration
            val = metrics.get(self.metric_key)
            if val is not None:
                self.best_metric = float(val)

        return record

    def _check_improvement(self, metrics: dict[str, Any]) -> bool:
        val = metrics.get(self.metric_key)
        if val is None:
            return False
        val = float(val)
        if self.best_metric is None:
            return True
        if self.metric_direction == "minimize":
            return val < self.best_metric
        return val > self.best_metric

    def summary(self) -> dict[str, Any]:
        return {
            "total_iterations": len(self.history),
            "best_iteration": self.best_iteration,
            "best_metric": self.best_metric,
            "metric_key": self.metric_key,
            "metric_direction": self.metric_direction,
            "all_metrics": [
                {"iteration": r.iteration, "metrics": r.metrics, "improved": r.improved}
                for r in self.history
            ],
        }
