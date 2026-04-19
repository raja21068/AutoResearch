"""Experiment agents: Planner, CodeGenerator, Runner, Tracker, Evaluator + Orchestrator."""

import json, logging, os, re, time, statistics, subprocess, tempfile
from pathlib import Path
from dataclasses import dataclass, field
from llm import llm

logger = logging.getLogger(__name__)


class ExperimentPlanner:
    SYSTEM = "You are an ML experiment planner. Design: hypothesis, variables, baselines, metrics, ablations, datasets."
    async def run(self, task, code="", data=""):
        return await llm(f"Task: {task}\nCode:\n{code[:2000]}\nData:\n{data[:800]}\nDesign experiment.", system=self.SYSTEM, agent="experiment")


class CodeGenerator:
    SYSTEM = "You are an ML engineer. Write a complete, runnable PyTorch training script. No placeholders."
    async def run(self, task, data="", metrics="accuracy"):
        raw = await llm(f"Task: {task}\nData:\n{data[:1000]}\nMetrics: {metrics}\nGenerate script.", system=self.SYSTEM, agent="coder")
        m = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()


class ExperimentRunner:
    """Runs scripts in sandbox or locally."""
    def __init__(self):
        from tools.sandbox import DockerSandbox
        self.sandbox = DockerSandbox()

    def run(self, script, timeout=600):
        stdout, stderr = self.sandbox.run_code(script)
        return {"success": not stderr.strip(), "stdout": stdout, "stderr": stderr}


@dataclass
class ExperimentRecord:
    id: str; hypothesis: str = ""; params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict); success: bool = False
    stdout: str = ""; elapsed: float = 0.0


class ResultTracker:
    """Tracks experiments to disk."""
    def __init__(self, base_dir="./output/tracker"):
        self.dir = Path(base_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.records: list[ExperimentRecord] = []

    def create(self, hypothesis="", params=None):
        r = ExperimentRecord(id=f"exp_{int(time.time()*1000)}", hypothesis=hypothesis, params=params or {})
        self.records.append(r); return r.id

    def log(self, exp_id, name, value):
        r = next((x for x in self.records if x.id == exp_id), None)
        if r: r.metrics.setdefault(name, []).append(value)

    def finalize(self, exp_id, success, stdout="", elapsed=0.0):
        r = next((x for x in self.records if x.id == exp_id), None)
        if not r: return ""
        r.success, r.stdout, r.elapsed = success, stdout[:5000], elapsed
        path = self.dir / f"{exp_id}.json"
        path.write_text(json.dumps(r.__dict__, indent=2, default=str)); return str(path)


class MetricsEvaluator:
    SYSTEM = "You are an ML analyst. Analyze metrics: findings, significance, baseline comparison, limitations."

    async def run(self, task, metrics=None, baseline=None):
        return await llm(f"Task: {task}\nMetrics:\n{metrics}\nBaseline:\n{baseline}\nAnalyze.",
                         system=self.SYSTEM, agent="experiment")

    @staticmethod
    def compute_stats(metrics, baseline=None):
        stats = {}
        for k, v in metrics.items():
            vals = [x["value"] if isinstance(x, dict) else float(x) for x in v] if isinstance(v, list) else [float(v)]
            s = {"mean": statistics.mean(vals), "n": len(vals)}
            if len(vals) > 1: s["std"] = statistics.stdev(vals)
            if baseline and k in baseline:
                b = float(baseline[k]) if not isinstance(baseline[k], list) else statistics.mean(baseline[k])
                s["improvement_pct"] = ((s["mean"] - b) / b * 100) if b else 0
            stats[k] = s
        return stats


class ExperimentOrchestrator:
    """End-to-end: plan → generate → run → track → evaluate."""
    def __init__(self):
        self.planner = ExperimentPlanner(); self.generator = CodeGenerator()
        self.runner = ExperimentRunner(); self.tracker = ResultTracker()
        self.evaluator = MetricsEvaluator()

    async def run_full(self, question, data=""):
        t0 = time.time()
        plan = await self.planner.run(question, data=data)
        script = await self.generator.run(question, data=data)
        exp_id = self.tracker.create(hypothesis=plan[:200])
        result = self.runner.run(script)
        self.tracker.finalize(exp_id, result["success"], result["stdout"], time.time()-t0)
        analysis = await self.evaluator.run(question, result.get("stdout",""))
        return {"plan": plan, "script": script, "result": result, "analysis": analysis, "elapsed": time.time()-t0}
