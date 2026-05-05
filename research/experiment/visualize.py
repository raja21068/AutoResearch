"""Experiment visualization — generate charts from experiment results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_all_charts(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    metric_key: str = "primary_metric",
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Generate matplotlib charts from experiment metrics across stages.

    Returns list of generated file paths.
    """
    if output_dir is None:
        output_dir = run_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []

    # Collect metrics from experiment stages
    all_metrics = _collect_metrics(run_dir)
    if not all_metrics:
        logger.warning("No experiment metrics found for chart generation")
        return generated

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping chart generation")
        return generated

    # -- Bar chart: final metrics per condition --
    try:
        conditions = list(all_metrics.keys())
        values = [m.get(metric_key, 0) for m in all_metrics.values()]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(conditions)), values, tick_label=conditions)
        ax.set_ylabel(metric_key)
        ax.set_title(f"{metric_key} by Condition")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        for fmt in formats:
            path = output_dir / f"comparison.{fmt}"
            fig.savefig(str(path), dpi=300)
            generated.append(str(path))
        plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to generate comparison chart: %s", exc)

    # -- Training curves (if iteration data available) --
    try:
        curves = _collect_training_curves(run_dir)
        if curves:
            fig, ax = plt.subplots(figsize=(10, 6))
            for label, points in curves.items():
                ax.plot(points, label=label)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Metric")
            ax.set_title("Training Curves")
            ax.legend()
            plt.tight_layout()

            for fmt in formats:
                path = output_dir / f"training_curves.{fmt}"
                fig.savefig(str(path), dpi=300)
                generated.append(str(path))
            plt.close(fig)
    except Exception as exc:
        logger.warning("Failed to generate training curves: %s", exc)

    logger.info("Generated %d charts in %s", len(generated), output_dir)
    return generated


def _collect_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Collect final metrics per condition from experiment output."""
    metrics: dict[str, dict[str, Any]] = {}
    for stage_name in ("stage-12", "stage-13"):
        stage_dir = run_dir / stage_name
        if not stage_dir.is_dir():
            continue
        for mf in sorted(stage_dir.glob("**/metrics*.json")):
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    label = mf.parent.name
                    metrics[label] = data
            except Exception:
                continue
    return metrics


def _collect_training_curves(run_dir: Path) -> dict[str, list[float]]:
    """Collect per-iteration metric values for training curve plots."""
    curves: dict[str, list[float]] = {}
    for stage_name in ("stage-12", "stage-13"):
        stage_dir = run_dir / stage_name
        if not stage_dir.is_dir():
            continue
        for log_file in sorted(stage_dir.glob("**/training_log*.jsonl")):
            try:
                label = log_file.parent.name
                points: list[float] = []
                for line in log_file.read_text(encoding="utf-8").splitlines():
                    entry = json.loads(line)
                    if "loss" in entry:
                        points.append(float(entry["loss"]))
                if points:
                    curves[label] = points
            except Exception:
                continue
    return curves
