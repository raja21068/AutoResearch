"""Engineering agents: Planner, Coder, Tester, Debugger, Critic."""

import json, logging, re
from typing import AsyncGenerator
from llm import llm, llm_stream

try:
    from unidiff import PatchSet
except ImportError:
    PatchSet = None

logger = logging.getLogger(__name__)


class PlannerAgent:
    SYSTEM = (
        "You are a senior software architect. Given a task, produce a JSON object with: "
        "'explanation', 'mode' (coding|research|hybrid), 'steps' (array of {agent, description, "
        "file?, parallel_group?, tool_name?, tool_params?}). "
        "ALWAYS include 'tester' after 'coder'. Return ONLY valid JSON."
    )

    async def create_plan(self, task, memory_context, repo_context) -> dict:
        raw = await llm(
            f"Task: {task}\nMemory:\n{memory_context}\nRepo:\n{repo_context}\nPlan now.",
            system=self.SYSTEM, agent="planner")
        try:
            return json.loads(raw)
        except Exception:
            return {"explanation": raw[:200], "mode": "coding",
                    "steps": [{"agent": "coder", "description": task}]}


class CoderAgent:
    SYSTEM_NEW = (
        "You are an expert software engineer. Write clean, production-ready code. "
        "Return ONLY code — no markdown fences, no explanations."
    )

    def _build_prompt(self, subtask, ctx_files, prev, memory, existing=""):
        ctx = "\n".join(ctx_files[:5]) if ctx_files else ""
        p = "\n".join(r.get("output","")[:300] for r in (prev or []) if isinstance(r, dict))
        if existing:
            return f"Existing:\n```python\n{existing}\n```\nTask: {subtask}\nContext:\n{ctx}\nMemory:\n{memory}\nGenerate diff."
        return f"Task: {subtask}\nContext:\n{ctx}\nPrevious:\n{p}\nMemory:\n{memory}\nWrite implementation."

    async def generate_code(self, subtask, ctx_files=None, prev=None, memory="", existing=""):
        raw = await llm(self._build_prompt(subtask, ctx_files or [], prev, memory, existing),
                        system=self.SYSTEM_NEW, agent="coder")
        return self._clean_code(raw)

    async def stream_code(self, subtask, ctx_files=None, prev=None, memory="", existing=""):
        async for tok in llm_stream(self._build_prompt(subtask, ctx_files or [], prev, memory, existing),
                                     system=self.SYSTEM_NEW, agent="coder"):
            yield tok

    @staticmethod
    def _clean_code(raw):
        if "```" in raw:
            m = re.search(r"```\w*\n(.*?)```", raw, re.DOTALL)
            if m: return m.group(1).strip()
        return raw.strip()


class TesterAgent:
    SYSTEM = (
        "You are a pytest expert. Generate self-contained test functions. "
        "Do NOT import from the code file — functions are already in scope. "
        "Return ONLY Python test functions, no markdown."
    )

    async def generate_tests(self, code: str, subtask: str) -> str:
        raw = await llm(
            f"Code:\n```python\n{code[:4000]}\n```\nTask: {subtask}\nWrite tests.",
            system=self.SYSTEM, agent="tester")
        return CoderAgent._clean_code(raw)


class DebuggerAgent:
    SYSTEM = (
        "You are an expert debugger. Given code and error, return the FIXED entire code. "
        "No explanation, no markdown."
    )

    async def fix(self, code: str, error: str) -> str:
        raw = await llm(
            f"Code:\n```python\n{code[:4000]}\n```\nError:\n{error[:2000]}\nFix it.",
            system=self.SYSTEM, agent="debugger")
        return CoderAgent._clean_code(raw)


class CriticAgent:
    SYSTEM = (
        "You are a senior reviewer. Evaluate pipeline output. "
        "Reply 'PASS' if correct or 'FAIL: reason'. Include score 1-10."
    )

    async def review(self, results: list[dict], task: str) -> str:
        summary = "\n".join(
            f"[{r.get('type','?')}] {r.get('step','')}: {str(r.get('output',''))[:400]}"
            for r in results)
        return await llm(f"Task: {task}\nResults:\n{summary}", system=self.SYSTEM, agent="critic")
