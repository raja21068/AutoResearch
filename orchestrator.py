"""
orchestrator.py — Brain OS: coordinates all agents.

Auto-fix loop: Coder → Tester → if fail → Debugger → retry (up to 3x).
Supports coding, research, and hybrid modes.
"""

import asyncio, json, logging, os, time
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

logger = logging.getLogger(__name__)

Callback = Callable[[str, str], Awaitable[None]] | None
MAX_DEBUG_RETRIES = 3


class Orchestrator:
    def __init__(self, repo_path=None):
        self.planner      = PlannerAgent()
        self.coder        = CoderAgent()
        self.tester       = TesterAgent()
        self.debugger     = DebuggerAgent()
        self.critic       = CriticAgent()
        self.memory       = MemoryAgent()
        self.researcher   = ResearcherAgent()
        self.experiment   = ExperimentAgent()
        self.paper_writer = PaperWriterAgent()
        self.sandbox      = DockerSandbox()
        self.tool_executor = ToolExecutor(cwd=repo_path or ".")
        self.output       = OutputManager(os.getenv("OUTPUT_DIR", "./output"))

    async def _emit(self, cb, evt, data):
        if cb: await cb(evt, data)

    async def run(self, task, context_files=None, callback=None,
                  papers_context="", repo_url="", dataset_context="", repo_changes=""):
        context_files = context_files or []
        t0 = time.time()
        run_out = self.output.create_run(task)

        # Build context
        full_ctx = ""
        if papers_context:  full_ctx += f"\n=== PAPERS ===\n{papers_context[:8000]}\n"
        if dataset_context: full_ctx += f"\n=== DATASET ===\n{dataset_context[:5000]}\n"
        if repo_changes:    full_ctx += f"\n=== MODIFICATIONS ===\n{repo_changes}\n"

        memory_ctx = self.memory.retrieve(task)
        await self._emit(callback, "agent", json.dumps({"agent":"planner","status":"running","step":"Planning"}))
        plan = await self.planner.create_plan(task + full_ctx, memory_ctx, "")
        await self._emit(callback, "plan", json.dumps({
            "explanation": plan.get("explanation",""), "mode": plan.get("mode","coding"),
            "steps": [{"agent":s.get("agent",""),"description":s.get("description","")} for s in plan.get("steps",[])]
        }))

        steps = plan.get("steps", [{"agent":"coder","description":task}])
        results, latest_code = [], ""

        for step in steps:
            agent = step.get("agent","coder")
            desc = step.get("description","")
            await self._emit(callback, "agent", json.dumps({"agent":agent,"status":"running","step":desc}))

            try:
                if agent == "coder":
                    r = await self._exec_coder(step, context_files, results, memory_ctx + full_ctx, callback)
                    if r.get("code"): latest_code = r["code"]
                elif agent == "tester":
                    r = await self._exec_tester(latest_code, desc, callback)
                elif agent == "tool":
                    tool = step.get("tool_name","")
                    out = self.tool_executor.execute(tool, step.get("tool_params",{}))
                    r = {"step":desc,"output":out.get("stdout",""),"type":"tool"}
                elif agent == "researcher":
                    data = await self.researcher.search_literature(desc)
                    r = {"step":desc,"output":data.get("review",""),"type":"research"}
                elif agent == "experiment":
                    out = await self.experiment.design_experiment(desc, latest_code)
                    r = {"step":desc,"output":out,"type":"experiment"}
                elif agent == "paper_writer":
                    res = "".join(x.get("output","")[:2000] for x in results if x.get("type")=="research")
                    exp = "".join(x.get("output","")[:2000] for x in results if x.get("type")=="experiment")
                    paper = await self.paper_writer.write_paper(desc, res, exp, latest_code[:500])
                    r = {"step":desc,"output":paper,"type":"paper"}
                elif agent == "critic":
                    r = {"step":desc,"output":await self.critic.review(results, desc),"type":"review"}
                elif agent == "debugger":
                    err = results[-1].get("output","") if results else ""
                    fixed = await self.debugger.fix(latest_code, err)
                    r = {"step":desc,"output":fixed,"type":"code","code":fixed}
                    latest_code = fixed
                else:
                    r = {"step":desc,"output":f"Unknown: {agent}","type":"unknown"}
            except Exception as e:
                await self._emit(callback, "error", json.dumps({"agent":agent,"error":str(e)}))
                r = {"step":desc,"output":str(e),"type":"error"}

            results.append(r)
            await self._emit(callback, "agent", json.dumps({"agent":agent,"status":"done","step":desc}))

        # Final review
        review = await self.critic.review(results, task)
        passed = "PASS" in review
        await self._emit(callback, "review", json.dumps({"review":review,"passed":passed}))

        # Save outputs
        elapsed = round(time.time()-t0, 2)
        if latest_code: run_out.save_code(latest_code)
        for i,r in enumerate(x for x in results if x.get("type")=="experiment"):
            run_out.save_experiment_results(r.get("output",""), f"exp_{i+1}.md")
        for r in results:
            if r.get("type")=="paper": run_out.save_paper(r.get("output",""))
        run_out.save_knowledge(f"Task: {task}\nPassed: {passed}\nReview: {review}")
        run_out.save_summary(results, passed, elapsed)
        if passed and latest_code: self.memory.store(task, latest_code[:2000])

        await self._emit(callback, "complete", json.dumps({"passed":passed,"steps":len(results),"elapsed_sec":elapsed}))
        return {"results":results,"passed":passed,"elapsed_sec":elapsed,"output_dir":str(run_out.run_dir)}

    async def _exec_coder(self, step, ctx_files, results, memory, cb):
        desc = step.get("description","")
        # Inject language rules
        if HAS_SKILLS:
            try:
                rules = get_rule_engine()
                for kw, lang in {"python":"python","java":"java","typescript":"typescript","go":"golang","rust":"rust"}.items():
                    if kw in desc.lower():
                        memory += "\n\n=== CODING RULES ===\n" + rules.get_rules(lang)[:2000]
                        break
            except Exception: pass

        code = ""
        async for token in self.coder.stream_code(desc, ctx_files, results, memory):
            code += token
            await self._emit(cb, "token", token)
        code = CoderAgent._clean_code(code) if "```" in code else code

        # Auto test→debug→fix loop
        for attempt in range(MAX_DEBUG_RETRIES):
            await self._emit(cb, "agent", json.dumps({"agent":"tester","status":"running","step":f"Test {attempt+1}/{MAX_DEBUG_RETRIES}"}))
            tests = await self.tester.generate_tests(code, desc)
            stdout, stderr = self.sandbox.run_code(code, tests)
            output = stdout + (f"\nSTDERR:\n{stderr}" if stderr else "")
            failed = bool(stderr.strip()) or "FAILED" in output or "Error" in output
            if not failed:
                await self._emit(cb, "test", json.dumps({"output":output[:1500],"passed":True}))
                break
            await self._emit(cb, "test", json.dumps({"output":output[:1500],"passed":False}))
            if attempt < MAX_DEBUG_RETRIES - 1:
                await self._emit(cb, "agent", json.dumps({"agent":"debugger","status":"running","step":f"Fix {attempt+1}"}))
                code = await self.debugger.fix(code, output)

        return {"step":desc,"output":code,"type":"code","code":code}

    async def _exec_tester(self, code, desc, cb):
        if not code: return {"step":desc,"output":"No code","type":"test"}
        tests = await self.tester.generate_tests(code, desc)
        stdout, stderr = self.sandbox.run_code(code, tests)
        return {"step":desc,"output":stdout+stderr,"type":"test"}

    async def run_streaming(self, task, context_files=None, **kwargs) -> AsyncGenerator[dict, None]:
        queue: asyncio.Queue = asyncio.Queue()
        async def _cb(evt, data): await queue.put({"event":evt,"data":data})
        async def _work():
            try: await self.run(task, context_files, callback=_cb, **kwargs)
            finally: await queue.put(None)
        worker = asyncio.create_task(_work())
        while True:
            item = await queue.get()
            if item is None: break
            yield item
        await worker

    def shutdown(self): pass
