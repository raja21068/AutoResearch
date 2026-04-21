"""
orchestrator.py — Brain OS: coordinates all agents.

Full pipeline:
  Input:  GitHub repos, PDFs, DOCX, datasets, instructions
  System: Planner → Coder → Tester → Debugger (K=3) → Critic
          Researcher → Experiment → FigureAgent → PaperWriter → Verification
  Output: code, experiments, figures, paper drafts, learned knowledge

Meta-learning: failures are recorded as lessons, successful patterns become reusable skills.
"""

import asyncio, json, logging, os, re, time
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable

from agents import (
    PlannerAgent, CoderAgent, TesterAgent, DebuggerAgent,
    CriticAgent, MemoryAgent, ResearcherAgent, ExperimentAgent, PaperWriterAgent,
)
from tools.sandbox import DockerSandbox
from tools.tool_executor import ToolExecutor
from tools.output_manager import OutputManager

try:
    from skills.engine import get_rule_engine
    HAS_SKILLS = True
except ImportError:
    HAS_SKILLS = False

try:
    from agents.verification import VerificationEngine
    HAS_VERIFICATION = True
except ImportError:
    HAS_VERIFICATION = False

try:
    from agents.paper import PaperOrchestrator, PlottingAgent
    HAS_PAPER = True
except ImportError:
    HAS_PAPER = False

logger = logging.getLogger(__name__)

Callback = Callable[[str, str], Awaitable[None]] | None
MAX_DEBUG_RETRIES = 3


class Orchestrator:
    """
    Central coordinator for the full AutoResearch pipeline.

    Manages: planning, coding, testing, debugging, reviewing,
    research, experiments, figures, paper writing, verification,
    and cross-run meta-learning.
    """

    def __init__(self, repo_path=None):
        # ── Core agents ──
        self.planner      = PlannerAgent()
        self.coder        = CoderAgent()
        self.tester       = TesterAgent()
        self.debugger     = DebuggerAgent()
        self.critic       = CriticAgent()
        self.memory       = MemoryAgent()
        self.researcher   = ResearcherAgent()
        self.experiment   = ExperimentAgent()
        self.paper_writer = PaperWriterAgent()

        # ── Tools ──
        self.sandbox       = DockerSandbox()
        self.tool_executor = ToolExecutor(cwd=repo_path or ".")
        self.output        = OutputManager(os.getenv("OUTPUT_DIR", "./output"))

        # ── Verification engine ──
        self.verifier = VerificationEngine() if HAS_VERIFICATION else None

        # ── Figure agent ──
        self.figure_agent = PlottingAgent() if HAS_PAPER else None

        # ── Paper orchestrator ──
        self.paper_orch = PaperOrchestrator() if HAS_PAPER else None

    # ── Helpers ──

    async def _emit(self, cb, evt, data):
        if cb:
            await cb(evt, data)

    def _classify_error(self, error_output: str) -> str:
        """Classify error type for meta-learning."""
        err = error_output.lower()
        if "syntaxerror" in err: return "syntax"
        if "typeerror" in err: return "type"
        if "indexerror" in err or "keyerror" in err: return "index_key"
        if "importerror" in err or "modulenotfounderror" in err: return "import"
        if "assertionerror" in err: return "assertion"
        if "timeout" in err or "timed out" in err: return "timeout"
        if "nameerror" in err: return "name"
        if "valueerror" in err: return "value"
        if "attributeerror" in err: return "attribute"
        if "runtimeerror" in err: return "runtime"
        return "logic"

    # ── Main entry ──

    async def run(self, task, context_files=None, callback=None,
                  papers_context="", repo_url="", dataset_context="",
                  repo_changes=""):
        context_files = context_files or []
        t0 = time.time()
        run_out = self.output.create_run(task)

        # ── Build context from all input types ──
        full_ctx = ""
        if papers_context:  full_ctx += f"\n=== PAPERS ===\n{papers_context[:8000]}\n"
        if dataset_context: full_ctx += f"\n=== DATASET ===\n{dataset_context[:5000]}\n"
        if repo_changes:    full_ctx += f"\n=== MODIFICATIONS ===\n{repo_changes}\n"
        if repo_url:        full_ctx += f"\n=== REPO ===\n{repo_url}\n"

        # ── Memory: retrieve past knowledge + meta-learning context ──
        memory_ctx = self.memory.get_context(task)

        # ── Plan ──
        await self._emit(callback, "agent", json.dumps(
            {"agent": "planner", "status": "running", "step": "Planning"}))
        plan = await self.planner.create_plan(task + full_ctx, memory_ctx, "")
        await self._emit(callback, "plan", json.dumps({
            "explanation": plan.get("explanation", ""),
            "mode": plan.get("mode", "coding"),
            "steps": [{"agent": s.get("agent", ""), "description": s.get("description", "")}
                      for s in plan.get("steps", [])]
        }))

        steps = plan.get("steps", [{"agent": "coder", "description": task}])
        results, latest_code = [], ""
        experiment_data = ""
        research_data = ""
        figures_data = ""

        # ── Execute each step ──
        for step in steps:
            agent = step.get("agent", "coder")
            desc = step.get("description", "")
            await self._emit(callback, "agent", json.dumps(
                {"agent": agent, "status": "running", "step": desc}))

            try:
                if agent == "coder":
                    r = await self._exec_coder(step, context_files, results,
                                                memory_ctx + full_ctx, callback)
                    if r.get("code"):
                        latest_code = r["code"]

                elif agent == "tester":
                    r = await self._exec_tester(latest_code, desc, callback)

                elif agent == "tool":
                    tool = step.get("tool_name", "")
                    out = self.tool_executor.execute(tool, step.get("tool_params", {}))
                    r = {"step": desc, "output": out.get("stdout", ""), "type": "tool"}

                elif agent == "researcher":
                    data = await self.researcher.search_literature(desc)
                    research_data = data.get("review", "")
                    r = {"step": desc, "output": research_data, "type": "research"}

                elif agent == "experiment":
                    out = await self.experiment.design_experiment(desc, latest_code)
                    experiment_data = out
                    r = {"step": desc, "output": out, "type": "experiment"}

                elif agent == "figure":
                    if self.figure_agent:
                        out = await self.figure_agent.run(
                            "", experiment_data[:3000], desc)
                        figures_data = out
                        r = {"step": desc, "output": out, "type": "figure"}
                    else:
                        r = {"step": desc, "output": "FigureAgent not available", "type": "figure"}

                elif agent == "paper_writer":
                    if self.paper_orch:
                        paper_result = await self.paper_orch.run_full(
                            desc, idea=full_ctx[:3000],
                            experiments=experiment_data[:3000])
                        paper_tex = paper_result.get("paper_tex", "")

                        # ── Verification ──
                        if self.verifier and paper_tex:
                            await self._emit(callback, "agent", json.dumps(
                                {"agent": "verifier", "status": "running",
                                 "step": "Verifying paper"}))
                            vr = await self.verifier.quick_check(paper_tex)
                            if not vr["passed"]:
                                logger.warning("Verification issues: %s", vr["issues"])
                            await self._emit(callback, "verification",
                                json.dumps({"passed": vr["passed"],
                                           "issues": vr.get("issues", [])}))

                        r = {"step": desc, "output": paper_tex, "type": "paper"}
                    else:
                        paper = await self.paper_writer.write_paper(
                            desc, research_data, experiment_data, latest_code[:500])
                        r = {"step": desc, "output": paper, "type": "paper"}

                elif agent == "critic":
                    r = {"step": desc,
                         "output": await self.critic.review(results, desc),
                         "type": "review"}

                elif agent == "debugger":
                    err = results[-1].get("output", "") if results else ""
                    fixed = await self.debugger.fix(latest_code, err)
                    r = {"step": desc, "output": fixed, "type": "code", "code": fixed}
                    latest_code = fixed

                else:
                    r = {"step": desc, "output": f"Unknown: {agent}", "type": "unknown"}

            except Exception as e:
                await self._emit(callback, "error",
                    json.dumps({"agent": agent, "error": str(e)}))
                r = {"step": desc, "output": str(e), "type": "error"}

            results.append(r)
            await self._emit(callback, "agent", json.dumps(
                {"agent": agent, "status": "done", "step": desc}))

        # ── Final review ──
        review = await self.critic.review(results, task)
        passed = "PASS" in review
        await self._emit(callback, "review",
            json.dumps({"review": review, "passed": passed}))

        # ── Save outputs ──
        elapsed = round(time.time() - t0, 2)
        if latest_code:
            run_out.save_code(latest_code)
        for i, r in enumerate(x for x in results if x.get("type") == "experiment"):
            run_out.save_experiment_results(r.get("output", ""), f"exp_{i+1}.md")
        for r in results:
            if r.get("type") == "paper":
                run_out.save_paper(r.get("output", ""))
            if r.get("type") == "figure":
                run_out.save_experiment_results(r.get("output", ""), "figures.tex")
        run_out.save_knowledge(f"Task: {task}\nPassed: {passed}\nReview: {review}")
        run_out.save_summary(results, passed, elapsed)

        # ── Meta-learning: store results + extract skills ──
        if passed and latest_code:
            self.memory.store(task, latest_code[:2000], success=True)
            # Extract reusable skill from successful pattern
            mode = plan.get("mode", "coding")
            self.memory.extract_skill(
                name=f"pattern_{mode}_{int(time.time())}",
                pattern=task[:100],
                solution=f"Plan mode: {mode}, {len(steps)} steps, "
                         f"agents: {','.join(s.get('agent','') for s in steps)}"
            )
        elif not passed:
            self.memory.store(task, f"FAILED: {review[:300]}", success=False)

        self.memory.save()

        await self._emit(callback, "complete", json.dumps({
            "passed": passed, "steps": len(results),
            "elapsed_sec": elapsed,
            "memory": self.memory.stats
        }))

        return {"results": results, "passed": passed,
                "elapsed_sec": elapsed, "output_dir": str(run_out.run_dir),
                "memory_stats": self.memory.stats}

    # ── Code execution with auto-fix loop + meta-learning ──

    async def _exec_coder(self, step, ctx_files, results, memory, cb):
        desc = step.get("description", "")

        # ── Inject language rules ──
        if HAS_SKILLS:
            try:
                rules = get_rule_engine()
                for kw, lang in {"python": "python", "java": "java",
                                  "typescript": "typescript", "go": "golang",
                                  "rust": "rust"}.items():
                    if kw in desc.lower():
                        memory += "\n\n=== CODING RULES ===\n" + rules.get_rules(lang)[:2000]
                        break
            except Exception:
                pass

        # ── Check memory for relevant fix suggestions ──
        past_fixes = self.memory.get_fix_suggestions("logic", desc[:50])
        if past_fixes:
            memory += "\n\n=== PAST FIX PATTERNS ===\n" + "\n".join(past_fixes[:3])

        # ── Generate code (streaming) ──
        code = ""
        async for token in self.coder.stream_code(desc, ctx_files, results, memory):
            code += token
            await self._emit(cb, "token", token)
        code = CoderAgent._clean_code(code) if "```" in code else code

        # ── Auto test→debug→fix loop ──
        for attempt in range(MAX_DEBUG_RETRIES):
            await self._emit(cb, "agent", json.dumps({
                "agent": "tester", "status": "running",
                "step": f"Test {attempt+1}/{MAX_DEBUG_RETRIES}"}))

            tests = await self.tester.generate_tests(code, desc)
            stdout, stderr = self.sandbox.run_code(code, tests)
            output = stdout + (f"\nSTDERR:\n{stderr}" if stderr else "")
            failed = bool(stderr.strip()) or "FAILED" in output or "Error" in output

            if not failed:
                await self._emit(cb, "test", json.dumps({
                    "output": output[:1500], "passed": True}))
                # ── Meta-learning: record success ──
                if attempt > 0:
                    self.memory.extract_skill(
                        name=f"fix_{desc[:30]}",
                        pattern=desc[:80],
                        solution=f"Fixed after {attempt} attempts"
                    )
                break

            await self._emit(cb, "test", json.dumps({
                "output": output[:1500], "passed": False}))

            # ── Meta-learning: record failure ──
            error_type = self._classify_error(output)

            if attempt < MAX_DEBUG_RETRIES - 1:
                await self._emit(cb, "agent", json.dumps({
                    "agent": "debugger", "status": "running",
                    "step": f"Fix {attempt+1}"}))

                # ── Inject past lessons for this error type ──
                lessons = self.memory.get_lessons(error_type, top_k=3)
                debug_ctx = output
                if lessons:
                    lesson_hints = "\n".join(
                        f"- Past fix for {l['error_type']}: {l['fix_applied'][:100]}"
                        for l in lessons if l["success"]
                    )
                    if lesson_hints:
                        debug_ctx += f"\n\n=== LEARNED FIXES ===\n{lesson_hints}"

                old_code = code
                code = await self.debugger.fix(code, debug_ctx)

                # Record the fix attempt
                self.memory.record_failure(
                    task=desc[:100], error_type=error_type,
                    error_msg=output[:200],
                    fix_applied=f"debugger_attempt_{attempt+1}",
                    fix_worked=False  # will update if next test passes
                )
            else:
                # Final failure — record lesson
                self.memory.record_failure(
                    task=desc[:100], error_type=error_type,
                    error_msg=output[:200],
                    fix_applied="exhausted_retries",
                    fix_worked=False
                )

        return {"step": desc, "output": code, "type": "code", "code": code}

    async def _exec_tester(self, code, desc, cb):
        if not code:
            return {"step": desc, "output": "No code", "type": "test"}
        tests = await self.tester.generate_tests(code, desc)
        stdout, stderr = self.sandbox.run_code(code, tests)
        return {"step": desc, "output": stdout + stderr, "type": "test"}

    # ── Streaming interface ──

    async def run_streaming(self, task, context_files=None, **kwargs) -> AsyncGenerator[dict, None]:
        queue: asyncio.Queue = asyncio.Queue()

        async def _cb(evt, data):
            await queue.put({"event": evt, "data": data})

        async def _work():
            try:
                await self.run(task, context_files, callback=_cb, **kwargs)
            finally:
                await queue.put(None)

        worker = asyncio.create_task(_work())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await worker

    def shutdown(self):
        self.memory.save()
