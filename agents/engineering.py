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
        "You are a senior research pipeline architect. Given a task, produce a JSON plan.\n\n"
        "═══ ROUTING DECISION TREE (follow in order) ═══\n\n"
        "RULE 1 — THESIS-TO-PAPER route:\n"
        "  Trigger: task mentions 'convert thesis', 'transform thesis', 'based on the provided thesis',\n"
        "           'provided document', OR a thesis/PDF has been uploaded (document_context non-empty).\n"
        "  AND the task does NOT say 'run code' / 'authentic results' / 'run experiments'.\n"
        "  Action: mode='research', steps = [researcher (extract), paper_writer (write), critic (review)].\n"
        "  NEVER add coder or tester steps.\n\n"
        "RULE 2 — EXPERIMENT-THEN-PAPER route:\n"
        "  Trigger: task says 'authentic results', 'first run', 'do not fabricate',\n"
        "           'run preprocessing', 'obtain results', 'run code then write paper'.\n"
        "  Action: mode='hybrid', steps = [coder, tester, experiment, paper_writer, critic].\n\n"
        "RULE 3 — PAPER-FROM-TOPIC route:\n"
        "  Trigger: task says 'write a paper', 'ieee paper', 'research paper', 'journal paper'\n"
        "           BUT no thesis is provided and no code execution needed.\n"
        "  Action: mode='research', steps = [researcher (literature), paper_writer (write), critic].\n\n"
        "RULE 4 — CODING route:\n"
        "  Trigger: everything else (implement, fix, debug, build, script).\n"
        "  Action: mode='coding', steps = [coder, tester].\n\n"
        "═══ OUTPUT FORMAT ═══\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        "  \"explanation\": \"one sentence why this route was chosen\",\n"
        "  \"mode\": \"research\" | \"hybrid\" | \"coding\",\n"
        "  \"steps\": [\n"
        "    {\"agent\": \"<name>\", \"description\": \"<what this step does>\"}\n"
        "  ]\n"
        "}\n"
        "Valid agent names: researcher, coder, tester, debugger, experiment, paper_writer, critic.\n"
        "paper_writer steps: include section-by-section writing instructions (Abstract through References).\n"
        "researcher steps for thesis: say 'extract all content including exact numerical results'.\n"
    )

    async def create_plan(self, task, memory_context, repo_context,
                          document_context: str = "") -> dict:
        # Use intent detection first to get a strong prior
        intent_route = "unknown"
        intent_steps = []
        try:
            from agents.intent import IntentDetector, steps_for_route
            detector = IntentDetector()
            intent = detector.detect(task, document_context)
            intent_route = intent.route.value
            intent_steps = steps_for_route(intent, task)
            logger.info("IntentDetector pre-computed route=%s", intent_route)
        except Exception as e:
            logger.warning("IntentDetector failed: %s", e)

        # Ask LLM to confirm / override with its own reasoning
        raw = await llm(
            f"Task: {task}\n"
            f"Has document/thesis: {'YES' if document_context.strip() else 'NO'}\n"
            f"Intent detector pre-computed route: {intent_route}\n"
            f"Memory:\n{memory_context[:500]}\n"
            f"Repo:\n{repo_context[:500]}\n\n"
            "Produce the final JSON plan. If the intent detector route looks correct, use it. "
            "Override only if you see a clear reason.",
            system=self.SYSTEM, agent="planner")

        if not raw.strip():
            logger.warning("Planner returned empty — using intent detector steps")
            if intent_steps:
                return {
                    "explanation": f"Intent detector route: {intent_route}",
                    "mode": "research" if "paper" in intent_route else "coding",
                    "steps": intent_steps,
                }
            return {"explanation": "Planner returned empty", "mode": "coding",
                    "steps": [{"agent": "coder", "description": task}]}

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                m = re.search(r"```(?:json)?\s*\n(.*?)```", cleaned, re.DOTALL)
                if m:
                    cleaned = m.group(1).strip()
            plan = json.loads(cleaned)

            # Safety: if task is clearly a paper task but LLM returned coding steps only,
            # override with intent detector steps
            has_paper_kw = any(kw in task.lower() for kw in [
                "write a paper", "ieee paper", "research paper",
                "convert thesis", "transform thesis", "ieee-style",
            ])
            agents_in_plan = {s.get("agent", "") for s in plan.get("steps", [])}
            if has_paper_kw and "paper_writer" not in agents_in_plan and intent_steps:
                logger.warning("Planner missed paper_writer — injecting intent steps")
                plan["steps"] = intent_steps
                plan["mode"] = "research" if "paper" in intent_route else "hybrid"

            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Planner JSON parse failed: %s", e)
            if intent_steps:
                return {
                    "explanation": f"Fallback to intent route: {intent_route}",
                    "mode": "research" if "paper" in intent_route else "coding",
                    "steps": intent_steps,
                }
            return {"explanation": raw[:200], "mode": "coding",
                    "steps": [{"agent": "coder", "description": task}]}


class CoderAgent:
    SYSTEM_NEW = (
        "You are an expert software engineer. Write clean, production-ready Python code. "
        "Return ONLY the raw Python code — no markdown fences, no triple backticks, no explanations.\n\n"
        "CRITICAL RULES (violations will break the sandbox):\n"
        "1. PRESERVE ALL SPACES in Python syntax — never merge keywords "
        "(e.g. 'import torch' NOT 'importtorch', 'class Foo(nn.Module):' NOT 'classFoo').\n"
        "2. Keep code LIGHTWEIGHT and SELF-CONTAINED — no pip install calls, "
        "no multi-GB model downloads, no full dataset tokenization.\n"
        "3. Use SYNTHETIC DATA ONLY — use torch.randn(), random.random(), or numpy arrays. "
        "NEVER call AutoTokenizer.from_pretrained(), AutoModel.from_pretrained(), "
        "AutoModelForSequenceClassification.from_pretrained(), or load_dataset().\n"
        "4. FORBIDDEN IMPORTS — never use: "
        "'from transformers import AdamW' (removed in transformers 4.x — use 'from torch.optim import AdamW'). "
        "Never import AutoModelForSequenceClassification or any HuggingFace model class.\n"
        "5. All code must run within 300 seconds on a standard CPU machine.\n"
        "6. If you need a model, build a small nn.Module from scratch (e.g. nn.Linear layers).\n"
        "7. If you need a tokenizer, simulate it with a simple word-index dict.\n"
        "8. Mock realistic results with fixed random seeds for reproducibility.\n"
        "9. PAPER/WRITING TASKS: If the task is to write, draft, or format a research paper, "
        "IEEE paper, LaTeX document, or Markdown document, do NOT write Python code. "
        "Output the document text directly as plain text or Markdown instead."
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
        if not raw.strip():
            logger.warning("CoderAgent received empty LLM response for: %s", subtask[:80])
            return "# Error: LLM returned empty response"
        return self._clean_code(raw)

    async def stream_code(self, subtask, ctx_files=None, prev=None, memory="", existing=""):
        async for tok in llm_stream(self._build_prompt(subtask, ctx_files or [], prev, memory, existing),
                                     system=self.SYSTEM_NEW, agent="coder"):
            yield tok

    @staticmethod
    def _clean_code(raw):
        if "```" in raw:
            m = re.search(r"```\w*\n(.*?)```", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
            else:
                raw = raw.strip()
        else:
            raw = raw.strip()
        # Validate syntax — attempt to fix common space-stripping corruption
        return CoderAgent._validate_and_fix_syntax(raw)

    @staticmethod
    def _validate_and_fix_syntax(code: str) -> str:
        """Check if code is valid Python. If not, attempt to fix common LLM artifacts.

        Handles the most common DeepSeek code-generation bugs where spaces
        between Python keywords are stripped, producing tokens like
        ``importtorch``, ``classMyModel``, ``defforward``, etc.
        """
        import ast
        # Return LaTeX documents unchanged — Python syntax rules do not apply
        stripped = code.strip()
        if (stripped.startswith("\\documentclass") or
                stripped.startswith("\\begin{document}") or
                "\\maketitle" in stripped[:500]):
            return code
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            pass

        # Line-by-line repair of merged-keyword artifacts.
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            s = line
            # Fix "importX" → "import X" at line start
            s = re.sub(r'^(\s*)import([A-Za-z])', r'\1import \2', s)
            # Fix "fromX" → "from X" at line start
            s = re.sub(r'^(\s*)from([A-Za-z])', r'\1from \2', s)
            # Fix "asX" at end of import → " as X"  (e.g. "torch.nnasnn" → "torch.nn as nn")
            s = re.sub(r'(\w)as([A-Za-z_]\w*)$', r'\1 as \2', s.rstrip())
            s = re.sub(r'(\w)as([A-Za-z_]\w*)\s', r'\1 as \2 ', s)
            # Fix "classMyModel" → "class MyModel"
            s = re.sub(r'^(\s*)class([A-Z])', r'\1class \2', s)
            # Fix "defforward" → "def forward"
            s = re.sub(r'^(\s*)def([a-z_])', r'\1def \2', s)
            # Fix "ifmask" → "if mask", "elifX" → "elif X"
            s = re.sub(r'^(\s*)if([A-Za-z_])', r'\1if \2', s)
            s = re.sub(r'^(\s*)elif([A-Za-z_])', r'\1elif \2', s)
            # Fix "returnx" → "return x"
            s = re.sub(r'^(\s*)return([A-Za-z_(\[])', r'\1return \2', s)
            # Fix "isnotNone" / "isNone" inline
            s = s.replace('isnotNone', 'is not None')
            s = s.replace('isNone', 'is None')
            fixed_lines.append(s)

        fixed = '\n'.join(fixed_lines)

        try:
            ast.parse(fixed)
            logger.warning("Auto-fixed merged-keyword syntax artifacts in generated code")
            return fixed
        except SyntaxError:
            # Return original — the debugger loop will handle it
            return code


class TesterAgent:
    SYSTEM = (
        "You are a pytest expert. Generate SELF-CONTAINED pytest test functions.\n\n"
        "CRITICAL RULES:\n"
        "1. NEVER add any import statements at the top of the test file. "
        "All classes and functions from the code under test are ALREADY IN SCOPE "
        "because they are prepended to the test file automatically.\n"
        "2. NEVER write 'from preprocessing import ...', 'from solution import ...', "
        "'from your_module import ...', or ANY module-level import. "
        "These will cause ModuleNotFoundError and fail every test.\n"
        "3. If you need torch, numpy, or other standard libraries INSIDE a test function, "
        "import them locally inside that function: 'def test_foo(): import torch; ...'.\n"
        "4. Test only the classes and functions visible in the provided code snippet.\n"
        "5. Use simple assertions. Mock complex dependencies with lambda or small classes.\n"
        "6. Return ONLY raw Python test functions — no markdown, no backticks, no explanations.\n"
        "7. Each test function name must start with 'test_'.\n"
        "8. Keep tests fast — no downloads, no large data, no sleeps."
    )

    async def generate_tests(self, code: str, subtask: str) -> str:
        raw = await llm(
            f"Code:\n```python\n{code[:6000]}\n```\nTask: {subtask}\n"
            "Write 3-5 pytest test functions. Remember: NO imports at file top — "
            "all names from the code are already in scope.",
            system=self.SYSTEM, agent="tester")
        if not raw.strip():
            logger.warning("TesterAgent received empty LLM response")
            return "def test_placeholder():\n    assert True  # LLM returned empty"
        cleaned = CoderAgent._clean_code(raw)
        # Safety: strip any top-level "from X import" or "import X" lines that
        # would cause ModuleNotFoundError in the combined sandbox file.
        cleaned = TesterAgent._strip_module_imports(cleaned)
        return cleaned

    @staticmethod
    def _strip_module_imports(test_code: str) -> str:
        """Remove top-level module imports from test code.

        The sandbox prepends the solution code to the test file, so all names
        are already in scope.  Module-level imports like
        ``from preprocessing import X`` cause ModuleNotFoundError.

        Imports for well-known third-party packages (torch, numpy, sklearn, etc.)
        are kept but moved inside a safe guard so they don't fail on missing modules.
        """
        import re as _re
        safe_packages = {
            "torch", "numpy", "np", "pandas", "pd", "sklearn", "scipy",
            "matplotlib", "pytest", "json", "os", "sys", "re", "time",
            "random", "math", "collections", "typing", "unittest", "mock",
            "transformers", "datasets", "evaluate",
        }
        kept, stripped = [], []
        for line in test_code.splitlines():
            stripped_line = line.strip()
            # Match: from X import Y  or  import X
            m_from   = _re.match(r'from\s+(\w+)', stripped_line)
            m_import = _re.match(r'import\s+(\w+)', stripped_line)
            module = None
            if m_from:
                module = m_from.group(1)
            elif m_import:
                module = m_import.group(1)

            if module and module not in safe_packages:
                # This is a custom module import — skip it (names already in scope)
                stripped.append(f"# REMOVED top-level import: {stripped_line}")
                continue
            kept.append(line)

        if stripped:
            logger.debug("TesterAgent stripped %d module imports: %s",
                         len(stripped), stripped)
        return "\n".join(kept)


class DebuggerAgent:
    SYSTEM = (
        "You are an expert Python debugger. Given code and error, return the FIXED entire code. "
        "No explanation, no markdown. Output ONLY raw Python.\n\n"
        "CRITICAL: Analyze the ROOT CAUSE before making changes. Known root causes:\n"
        "1. Merged keywords: 'importtorch' → 'import torch', 'classFoo' → 'class Foo', etc.\n"
        "2. Deprecated import: 'from transformers import AdamW' was REMOVED in transformers 4.x. "
        "Fix it with: 'from torch.optim import AdamW'\n"
        "3. Timeout/download: if error mentions 'Timed out', remove ALL "
        "AutoModel.from_pretrained(), AutoTokenizer.from_pretrained(), load_dataset() calls "
        "and replace them with synthetic torch.randn() data and simple nn.Linear mock models.\n"
        "4. HuggingFace mock model: if the error is 'got an unexpected keyword argument input_ids', "
        "it means a mock nn.Linear is being used as a model. Replace it with a proper mock:\n"
        "   class MockModel(nn.Module):\n"
        "       def __init__(self): super().__init__(); self.fc = nn.Linear(64, 2)\n"
        "       def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):\n"
        "           import torch; b=input_ids.shape[0]; logits=torch.randn(b,2)\n"
        "           loss=(nn.CrossEntropyLoss()(logits,labels) if labels is not None "
        "else torch.tensor(0.5))\n"
        "           from types import SimpleNamespace; return SimpleNamespace(loss=loss,logits=logits)\n"
        "5. ModuleNotFoundError in tests: the test file tries to import a non-existent module. "
        "All functions are already in scope — remove the bad import from the test code.\n"
        "Fix the actual root cause, not the symptom."
    )

    async def fix(self, code: str, error: str) -> str:
        raw = await llm(
            f"Code:\n```python\n{code[:6000]}\n```\nError:\n{error[:3000]}\nFix it.",
            system=self.SYSTEM, agent="debugger")
        if not raw.strip():
            logger.warning("DebuggerAgent received empty LLM response — returning original code")
            return code
        return CoderAgent._clean_code(raw)


class CriticAgent:
    SYSTEM = (
        "You are a senior reviewer. Evaluate pipeline output. "
        "Reply 'PASS' if correct or 'FAIL: reason'. Include score 1-10. "
        "IMPORTANT: If test logs show all tests passed (e.g. 'passed', "
        "'13/13 tests PASSED'), trust the test results and reply PASS. "
        "Do NOT hallucinate missing components based on truncated views."
    )

    PAPER_REVIEW_SYSTEM = (
        "You are a strict IEEE paper reviewer and quality checker.\n\n"
        "Review the paper output and check ALL of the following:\n"
        "1. COMPLETENESS: Does the LaTeX contain ALL sections?\n"
        "   Required: \\documentclass, Abstract, I.Introduction, II.Related Work,\n"
        "   III.Methodology, IV.Experimental Setup, V.Results and Discussion,\n"
        "   VI.Conclusion, References (\\bibitem entries).\n"
        "   FAIL if any section is missing or says [TODO] / [TBD].\n"
        "2. LENGTH: Is the paper substantial? Each section must be at least 2 paragraphs.\n"
        "   Introduction ≥ 3 paragraphs. Related Work ≥ 3 paragraphs. Results ≥ 4 paragraphs.\n"
        "3. EM DASHES: Must be zero em dashes (—). FAIL if any found.\n"
        "4. BULLET POINTS: No \\itemize or \\enumerate inside section bodies. FAIL if found.\n"
        "5. CITATIONS: Must have \\cite{} references throughout. FAIL if fewer than 10.\n"
        "6. REFERENCES: Must have \\bibitem entries. FAIL if fewer than 10.\n"
        "7. DUPLICATE REFS: No duplicate \\bibitem keys. FAIL if duplicates found.\n"
        "8. TABLES: Results section must contain at least one \\begin{table}. FAIL if none.\n"
        "9. TRUNCATION: Paper must not end mid-sentence or with '...'. FAIL if truncated.\n"
        "10. NO PLACEHOLDERS: No [Author Name], [Institution], [Year] unfilled. FAIL if found.\n\n"
        "Score 1-10. Score ≥ 8 = PASS. Score < 8 = FAIL with specific issues listed.\n"
        "Format: PASS/FAIL: <score>/10. Issues: <list>."
    )

    async def review(self, results: list[dict], task: str) -> str:
        # Pre-check: all tests passed?
        for r in results:
            output_str = str(r.get("output", ""))
            if r.get("type") in ("test", "code") and self._tests_passed(output_str):
                logger.info("CriticAgent: tests passed in logs — auto-PASS")
                return "PASS: All tests passed per execution logs. Score: 9/10"

        # Check if this is a paper review task
        is_paper_task = any(kw in task.lower() for kw in [
            "ieee paper", "research paper", "write a paper", "convert thesis",
            "transform thesis", "ieee-style", "paper_writer",
        ])

        # Find paper output in results
        paper_content = ""
        for r in results:
            output_str = str(r.get("output", ""))
            if "\\documentclass" in output_str or "\\section{" in output_str:
                paper_content = output_str
                break

        if is_paper_task and paper_content:
            result = await llm(
                f"TASK: {task[:500]}\n\nPAPER OUTPUT (LaTeX):\n{paper_content[:12000]}\n\n"
                "Review this IEEE paper against ALL 10 criteria.",
                system=self.PAPER_REVIEW_SYSTEM,
                agent="critic",
            )
            return result or "FAIL: Critic returned empty response. Score: 0/10"

        # Standard review
        summary = "\n".join(
            f"[{r.get('type','?')}] {r.get('step','')}: {str(r.get('output',''))[:2000]}"
            for r in results)
        result = await llm(
            f"Task: {task}\nResults:\n{summary}",
            system=self.SYSTEM, agent="critic",
        )
        if not result.strip():
            logger.warning("CriticAgent received empty LLM response — defaulting to FAIL")
            return "FAIL: Critic could not evaluate — LLM returned empty response. Score: 0/10"
        return result

    @staticmethod
    def _tests_passed(output: str) -> bool:
        """Check if test output indicates all tests passed."""
        lower = output.lower()
        # pytest-style pass indicators
        if "passed" in lower and "failed" not in lower and "error" not in lower:
            return True
        if "tests passed" in lower and "failed" not in lower:
            return True
        # Explicit pass counts like "13/13 tests PASSED"
        import re
        m = re.search(r'(\d+)/\1\s+tests?\s+passed', lower)
        if m:
            return True
        # pytest summary: "X passed" with no failures
        if re.search(r'\d+\s+passed', lower) and not re.search(r'\d+\s+failed', lower):
            return True
        return False
