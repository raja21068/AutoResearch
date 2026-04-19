"""Unified code generation agent for experiment Stages 10 & 13.

Merges two previously separate implementations into one coherent module:

  - ``experiment/code_agent.py``  — pluggable provider backends
    (LLM chat, Claude Code CLI, Codex CLI)
  - ``pipeline/code_agent.py``    — advanced multi-phase CodeAgent
    (blueprint → sequential generation → exec-fix → tree search → review)

All providers now share a single ``CodeAgentResult`` and implement the
same ``CodeAgentProvider`` protocol, so any provider can be used anywhere.

Providers
---------
``"llm"``       — existing LLM chat API (backward-compatible default)
``"advanced"``  — multi-phase CodeAgent (blueprint, tree search, review)
``"claude_code"``— Claude Code CLI (``claude -p``)
``"codex"``     — OpenAI Codex CLI (``codex exec``)

Usage::

    from research.experiment.code_agent import create_code_agent

    agent = create_code_agent(config, llm=llm_client, prompts=pm)
    result = agent.generate(exp_plan=plan, topic=topic, ...)
    if result.ok:
        files = result.files      # dict[str, str]
        spec  = result.architecture_spec  # "" for non-advanced providers
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from research.config import RCConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CodeAgentResult:
    """Output from any code agent invocation.

    Fields used by all providers
    ----------------------------
    files         dict[str, str]  filename → source code
    provider_name str             "llm" | "advanced" | "claude_code" | "codex"
    elapsed_sec   float           wall-clock time
    ok            bool (property) True when files is non-empty and error is None

    Fields populated only by the advanced provider
    -----------------------------------------------
    architecture_spec    YAML blueprint produced during Phase 1
    validation_log       step-by-step generation log
    total_llm_calls      LLM calls consumed
    total_sandbox_runs   sandbox executions consumed
    best_score           score of the winning solution node
    tree_nodes_explored  nodes visited during tree search (0 if disabled)
    review_rounds        coder-reviewer dialog rounds completed
    """

    files: dict[str, str]
    provider_name: str
    elapsed_sec: float
    # Basic providers
    raw_output: str = ""
    error: str | None = None
    # Advanced provider extras (default to safe zero-values)
    architecture_spec: str = ""
    validation_log: list[str] = field(default_factory=list)
    total_llm_calls: int = 0
    total_sandbox_runs: int = 0
    best_score: float = 0.0
    tree_nodes_explored: int = 0
    review_rounds: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.files)


# ---------------------------------------------------------------------------
# Advanced agent configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeAgentConfig:
    """Configuration for the advanced multi-phase code generation agent.

    All phases are independently toggleable.  The default profile enables
    Phases 1 (blueprint), 2 (sequential generation + exec-fix), and 5
    (review).  Phase 4 (tree search) is opt-in due to higher cost.
    """

    enabled: bool = True

    # Phase 1: Blueprint planning
    architecture_planning: bool = True

    # Phase 2: Sequential file generation
    sequential_generation: bool = True

    # Phase 2.5: Hard validation gates (AST-based)
    hard_validation: bool = True
    hard_validation_max_repairs: int = 4

    # Phase 3: Execution-in-the-loop
    exec_fix_max_iterations: int = 3
    exec_fix_timeout_sec: int = 60

    # Phase 4: Solution tree search (off by default)
    tree_search_enabled: bool = False
    tree_search_candidates: int = 3
    tree_search_max_depth: int = 2
    tree_search_eval_timeout_sec: int = 120

    # Phase 5: Multi-agent review dialog
    review_max_rounds: int = 2


# ---------------------------------------------------------------------------
# Solution node (tree search)
# ---------------------------------------------------------------------------


@dataclass
class SolutionNode:
    """One candidate solution in the tree search."""

    node_id: str
    files: dict[str, str]
    parent_id: str | None = None
    depth: int = 0
    runs_ok: bool = False
    returncode: int = -1
    evaluated: bool = False
    stdout: str = ""
    stderr: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    generation_method: str = "initial"


# ---------------------------------------------------------------------------
# Structural protocols
# ---------------------------------------------------------------------------


class CodeAgentProvider(Protocol):  # pragma: no cover
    """Implemented by every code generation backend."""

    @property
    def name(self) -> str: ...

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult: ...

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult: ...

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult: ...


class _SandboxResult(Protocol):  # pragma: no cover
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    metrics: dict[str, object]
    timed_out: bool


class _SandboxLike(Protocol):  # pragma: no cover
    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _collect_py_files(workdir: Path) -> dict[str, str]:
    """Read all .py files from a directory (flat, no subdirs)."""
    files: dict[str, str] = {}
    for pyfile in sorted(workdir.glob("*.py")):
        if pyfile.name.startswith(("_codex_", "_agent_")):
            continue
        files[pyfile.name] = pyfile.read_text(encoding="utf-8")
    return files


def _seed_workdir(workdir: Path, files: dict[str, str]) -> None:
    """Pre-populate workdir with files for refinement/repair."""
    workdir.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (workdir / fname).write_text(content, encoding="utf-8")


def _extract_files_from_text(content: str) -> dict[str, str]:
    """Extract multi-file code blocks from LLM output.

    Imports from pipeline._helpers (not pipeline.executor) to avoid
    the circular import that previously existed in pipeline/code_agent.py.
    """
    from research.pipeline._helpers import _extract_multi_file_blocks
    return _extract_multi_file_blocks(content)


def _extract_single_block(content: str) -> str:
    """Extract a single code block from LLM output."""
    from research.pipeline._helpers import _extract_code_block
    return _extract_code_block(content)


def format_feedback_for_agent(
    sandbox_result: Any,
    metric_key: str,
    metric_direction: str,
    best_metric: float | None,
) -> str:
    """Format sandbox run results as structured feedback for CLI agents."""
    parts = ["## Previous Run Results"]
    parts.append(f"Return code: {sandbox_result.returncode}")
    parts.append(f"Elapsed: {sandbox_result.elapsed_sec:.1f}s")
    parts.append(f"Timed out: {sandbox_result.timed_out}")
    if sandbox_result.metrics:
        parts.append("Metrics:")
        for k, v in sandbox_result.metrics.items():
            parts.append(f"  {k}: {v}")
    if sandbox_result.stderr:
        parts.append(f"Stderr (last 1000 chars):\n{sandbox_result.stderr[-1000:]}")
    parts.append(f"\nTarget: {metric_direction} '{metric_key}'")
    if best_metric is not None:
        parts.append(f"Best so far: {best_metric}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LlmCodeAgent — wraps the existing LLM chat API
# ---------------------------------------------------------------------------


class LlmCodeAgent:
    """Code agent backed by the OpenAI-compatible LLM chat API.

    Backward-compatible default.  Extracts and preserves the LLM call
    logic that was previously inline in ``_execute_code_generation`` and
    ``_execute_iterative_refine``.
    """

    def __init__(self, llm: Any, prompts: Any, config: RCConfig) -> None:
        self._llm = llm
        self._pm = prompts
        self._config = config

    @property
    def name(self) -> str:
        return "llm"

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        from research.pipeline._helpers import _chat_with_prompt

        start = time.monotonic()
        sp = self._pm.for_stage(
            "code_generation",
            topic=topic,
            metric=metric_key,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            exp_plan=exp_plan,
        )
        _max_tokens = sp.max_tokens or 8192
        if any(
            self._config.llm.primary_model.startswith(p)
            for p in ("gpt-5", "o3", "o4")
        ):
            _max_tokens = max(_max_tokens, 16384)

        try:
            resp = _chat_with_prompt(
                self._llm, sp.system, sp.user,
                json_mode=sp.json_mode, max_tokens=_max_tokens,
            )
            files = _extract_files_from_text(resp.content)
            if not files and not resp.content.strip():
                logger.warning(
                    "LlmCodeAgent: empty response — retrying with 32768 tokens"
                )
                resp = _chat_with_prompt(
                    self._llm, sp.system, sp.user,
                    json_mode=sp.json_mode, max_tokens=32768,
                )
                files = _extract_files_from_text(resp.content)

            return CodeAgentResult(
                files=files,
                provider_name="llm",
                elapsed_sec=time.monotonic() - start,
                raw_output=resp.content[:2000],
            )
        except Exception as exc:
            logger.error("LlmCodeAgent.generate failed: %s", exc)
            return CodeAgentResult(
                files={}, provider_name="llm",
                elapsed_sec=time.monotonic() - start, error=str(exc),
            )

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        from research.pipeline._helpers import _chat_with_prompt

        start = time.monotonic()

        def _files_ctx(pf: dict[str, str]) -> str:
            return "\n\n".join(
                f"```filename:{f}\n{c}\n```" for f, c in sorted(pf.items())
            )

        try:
            ip = self._pm.sub_prompt(
                "iterative_improve",
                metric_key=metric_key,
                metric_direction=metric_direction,
                files_context=_files_ctx(current_files),
                run_summaries="\n".join(run_summaries[:20]),
                condition_coverage_hint="",
                topic=topic,
            )
            response = _chat_with_prompt(
                self._llm, ip.system, ip.user + extra_hints,
                max_tokens=ip.max_tokens or 8192,
            )
            extracted = _extract_files_from_text(response.content)
            if not extracted:
                code = _extract_single_block(response.content)
                if code.strip():
                    extracted = {"main.py": code}
            return CodeAgentResult(
                files=extracted, provider_name="llm",
                elapsed_sec=time.monotonic() - start,
                raw_output=response.content[:2000],
            )
        except Exception as exc:
            logger.error("LlmCodeAgent.refine failed: %s", exc)
            return CodeAgentResult(
                files={}, provider_name="llm",
                elapsed_sec=time.monotonic() - start, error=str(exc),
            )

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        from research.pipeline._helpers import _chat_with_prompt

        start = time.monotonic()
        all_files_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        try:
            rp = self._pm.sub_prompt(
                "code_repair",
                fname="main.py",
                issues_text=issues,
                all_files_ctx=all_files_ctx,
            )
            resp = _chat_with_prompt(self._llm, rp.system, rp.user)
            repaired = _extract_files_from_text(resp.content)
            if not repaired:
                code = _extract_single_block(resp.content)
                if code.strip():
                    repaired = {"main.py": code}
            return CodeAgentResult(
                files=repaired, provider_name="llm",
                elapsed_sec=time.monotonic() - start,
                raw_output=resp.content[:2000],
            )
        except Exception as exc:
            return CodeAgentResult(
                files={}, provider_name="llm",
                elapsed_sec=time.monotonic() - start, error=str(exc),
            )


# ---------------------------------------------------------------------------
# CLI agent base — shared subprocess logic for Claude Code / Codex
# ---------------------------------------------------------------------------


class _CliAgentBase:
    """Shared subprocess infrastructure for CLI-based coding agents."""

    _provider_name: str = ""

    def __init__(
        self,
        binary_path: str,
        model: str = "",
        max_budget_usd: float = 5.0,
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
    ) -> None:
        self._binary = binary_path
        self._model = model
        self._max_budget_usd = max_budget_usd
        self._default_timeout = timeout_sec
        self._extra_args = extra_args or []

    @property
    def name(self) -> str:
        return self._provider_name

    def _run_subprocess(
        self, cmd: list[str], workdir: Path, timeout_sec: int,
    ) -> tuple[int, str, str, float, bool]:
        """Run command as subprocess with process-group cleanup on timeout."""
        workdir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        timed_out = False
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=workdir, env={**os.environ}, start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)

        return (
            proc.returncode or -1,
            _to_text(stdout_bytes), _to_text(stderr_bytes),
            time.monotonic() - start, timed_out,
        )

    def _build_result(
        self, workdir: Path, returncode: int,
        stdout: str, stderr: str, elapsed: float, timed_out: bool,
    ) -> CodeAgentResult:
        files = _collect_py_files(workdir)
        error = None
        if timed_out:
            error = f"Timed out after {elapsed:.0f}s"
        elif returncode != 0 and not files:
            error = f"Exited {returncode}: {stderr[:500]}"
        return CodeAgentResult(
            files=files, provider_name=self._provider_name,
            elapsed_sec=elapsed, raw_output=stdout[:3000], error=error,
        )

    # -- Shared prompt builders -------------------------------------------

    @staticmethod
    def _generate_prompt(
        topic: str, exp_plan: str, metric_key: str,
        pkg_hint: str, compute_budget: str, extra_guidance: str,
    ) -> str:
        return (
            "You are generating experiment code for a research paper.\n\n"
            f"TOPIC: {topic}\n\n"
            f"EXPERIMENT PLAN:\n{exp_plan}\n\n"
            f"PRIMARY METRIC: {metric_key}\n"
            f"{pkg_hint}\n{compute_budget}\n{extra_guidance}\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a multi-file Python project in the current directory.\n"
            "2. The entry point MUST be main.py.\n"
            "3. main.py must print metrics as 'name: value' lines to stdout.\n"
            f"4. Use condition labels: 'condition=<n> {metric_key}: <value>'\n"
            "5. FORBIDDEN: subprocess, os.system, eval, exec, shutil, socket, "
            "network calls, external data files.\n"
            "6. Use deterministic seeds (numpy.random.seed or random.seed).\n"
            "7. Write ALL files to the current working directory.\n"
            "8. Do NOT create subdirectories.\n"
        )

    @staticmethod
    def _refine_prompt(
        current_files: dict[str, str], run_summaries: list[str],
        metric_key: str, metric_direction: str, topic: str, extra_hints: str,
    ) -> str:
        files_listing = "\n".join(
            f"  - {f} ({len(c)} chars)" for f, c in current_files.items()
        )
        summaries_text = "\n".join(run_summaries[:10]) or "(no prior runs)"
        return (
            "You are improving experiment code for a research paper.\n\n"
            f"TOPIC: {topic}\nTARGET: {metric_direction} '{metric_key}'\n\n"
            f"EXISTING FILES:\n{files_listing}\n\n"
            f"PRIOR RUN SUMMARIES:\n{summaries_text}\n\n{extra_hints}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read existing code and understand the experiment structure.\n"
            "2. Modify files to improve the metric.\n"
            "3. Keep the entry point as main.py.\n"
            "4. Write modified files to the current directory.\n"
            "5. FORBIDDEN: subprocess, os.system, eval, exec, shutil, socket.\n"
        )

    @staticmethod
    def _repair_prompt(files: dict[str, str], issues: str) -> str:
        files_listing = "\n".join(
            f"  - {f} ({len(c)} chars)" for f, c in files.items()
        )
        return (
            "The experiment code has validation or runtime issues.\n\n"
            f"ISSUES:\n{issues}\n\nFILES:\n{files_listing}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the existing files in the current directory.\n"
            "2. Fix ALL reported issues.\n"
            "3. Write the corrected files back.\n"
            "4. FORBIDDEN: subprocess, os.system, eval, exec, shutil, socket.\n"
        )


# ---------------------------------------------------------------------------
# ClaudeCodeAgent — Claude Code CLI backend
# ---------------------------------------------------------------------------


class ClaudeCodeAgent(_CliAgentBase):
    """Code agent backed by the Claude Code CLI (``claude -p``)."""

    _provider_name = "claude_code"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary, "-p", prompt,
            "--dangerously-skip-permissions",
            "--output-format", "text",
            "--allowed-tools", "Bash Edit Write Read",
            "--add-dir", str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        if self._max_budget_usd:
            cmd += ["--max-budget-usd", str(self._max_budget_usd)]
        cmd.extend(self._extra_args)
        return cmd

    def generate(self, *, exp_plan: str, topic: str, metric_key: str,
                 pkg_hint: str, compute_budget: str, extra_guidance: str,
                 workdir: Path, timeout_sec: int = 600) -> CodeAgentResult:
        cmd = self._build_cmd(
            self._generate_prompt(topic, exp_plan, metric_key, pkg_hint, compute_budget, extra_guidance),
            workdir,
        )
        rc, out, err, elapsed, to = self._run_subprocess(cmd, workdir, timeout_sec or self._default_timeout)
        return self._build_result(workdir, rc, out, err, elapsed, to)

    def refine(self, *, current_files: dict[str, str], run_summaries: list[str],
               metric_key: str, metric_direction: str, topic: str, extra_hints: str,
               workdir: Path, timeout_sec: int = 600) -> CodeAgentResult:
        _seed_workdir(workdir, current_files)
        cmd = self._build_cmd(
            self._refine_prompt(current_files, run_summaries, metric_key, metric_direction, topic, extra_hints),
            workdir,
        )
        rc, out, err, elapsed, to = self._run_subprocess(cmd, workdir, timeout_sec or self._default_timeout)
        return self._build_result(workdir, rc, out, err, elapsed, to)

    def repair(self, *, files: dict[str, str], issues: str,
               workdir: Path, timeout_sec: int = 300) -> CodeAgentResult:
        _seed_workdir(workdir, files)
        cmd = self._build_cmd(self._repair_prompt(files, issues), workdir)
        rc, out, err, elapsed, to = self._run_subprocess(cmd, workdir, timeout_sec or self._default_timeout)
        return self._build_result(workdir, rc, out, err, elapsed, to)


# ---------------------------------------------------------------------------
# CodexAgent — OpenAI Codex CLI backend
# ---------------------------------------------------------------------------


class CodexAgent(_CliAgentBase):
    """Code agent backed by the OpenAI Codex CLI (``codex exec``)."""

    _provider_name = "codex"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary, "exec", prompt,
            "--sandbox", "workspace-write",
            "--json", "-C", str(workdir),
        ]
        if self._model:
            cmd += ["-m", self._model]
        cmd.extend(self._extra_args)
        return cmd

    def generate(self, *, exp_plan: str, topic: str, metric_key: str,
                 pkg_hint: str, compute_budget: str, extra_guidance: str,
                 workdir: Path, timeout_sec: int = 600) -> CodeAgentResult:
        cmd = self._build_cmd(
            self._generate_prompt(topic, exp_plan, metric_key, pkg_hint, compute_budget, extra_guidance),
            workdir,
        )
        rc, out, err, elapsed, to = self._run_subprocess(cmd, workdir, timeout_sec or self._default_timeout)
        return self._build_result(workdir, rc, out, err, elapsed, to)

    def refine(self, *, current_files: dict[str, str], run_summaries: list[str],
               metric_key: str, metric_direction: str, topic: str, extra_hints: str,
               workdir: Path, timeout_sec: int = 600) -> CodeAgentResult:
        _seed_workdir(workdir, current_files)
        cmd = self._build_cmd(
            self._refine_prompt(current_files, run_summaries, metric_key, metric_direction, topic, extra_hints),
            workdir,
        )
        rc, out, err, elapsed, to = self._run_subprocess(cmd, workdir, timeout_sec or self._default_timeout)
        return self._build_result(workdir, rc, out, err, elapsed, to)

    def repair(self, *, files: dict[str, str], issues: str,
               workdir: Path, timeout_sec: int = 300) -> CodeAgentResult:
        _seed_workdir(workdir, files)
        cmd = self._build_cmd(self._repair_prompt(files, issues), workdir)
        rc, out, err, elapsed, to = self._run_subprocess(cmd, workdir, timeout_sec or self._default_timeout)
        return self._build_result(workdir, rc, out, err, elapsed, to)


# ---------------------------------------------------------------------------
# CodeAgent — advanced multi-phase provider (formerly pipeline/code_agent.py)
# ---------------------------------------------------------------------------


class CodeAgent:
    """Multi-phase code generation agent — now implements CodeAgentProvider.

    Phases
    ------
    1. Blueprint planning  — deep YAML blueprint with per-file pseudocode
    2. Sequential generation — file-by-file following blueprint order
    2.5 Hard validation    — AST-based gates with auto-repair
    3. Exec-fix loop       — run in sandbox, feed errors back for repair
    4. Tree search         — explore N candidates, evaluate, keep best (opt-in)
    5. Review dialog       — coder-reviewer rounds for quality assurance

    Implements ``generate``, ``refine``, and ``repair`` so it can be used
    anywhere a ``CodeAgentProvider`` is expected.
    """

    _provider_name = "advanced"

    def __init__(
        self,
        llm: Any,
        prompts: Any,
        config: CodeAgentConfig,
        stage_dir: Path,
        sandbox_factory: Any | None = None,
        experiment_config: Any | None = None,
        domain_profile: Any | None = None,
        code_search_result: Any | None = None,
    ) -> None:
        self._llm = llm
        self._pm = prompts
        self._cfg = config
        self._stage_dir = stage_dir
        self._sandbox_factory = sandbox_factory
        self._exp_config = experiment_config
        self._domain_profile = domain_profile
        self._code_search_result = code_search_result
        self._calls = 0
        self._runs = 0
        self._log: list[str] = []
        self._sandbox: _SandboxLike | None = None

    @property
    def name(self) -> str:
        return self._provider_name

    # -- CodeAgentProvider: generate --------------------------------------- #

    def generate(
        self,
        *,
        exp_plan: str,
        topic: str,
        metric_key: str,
        pkg_hint: str,
        compute_budget: str,
        extra_guidance: str,
        workdir: Path,
        timeout_sec: int = 600,
        # Advanced-only kwargs (used when called directly from _code_generation.py)
        metric: str = "",
        max_tokens: int = 8192,
    ) -> CodeAgentResult:
        """Execute all enabled phases and return generated files."""
        # Allow legacy positional-style calls from _code_generation.py
        _metric = metric or metric_key
        _pkg = pkg_hint + "\n" + compute_budget + "\n" + extra_guidance

        t0 = time.time()
        self._log_event("CodeAgent.generate() started")

        arch_spec = ""
        blueprint = None
        if self._cfg.architecture_planning:
            arch_spec, blueprint = self._phase1_blueprint(topic, exp_plan, _metric)

        nodes_explored = 0
        if self._cfg.tree_search_enabled and self._sandbox_factory:
            best, nodes_explored = self._phase3_tree_search(
                topic, exp_plan, _metric, _pkg, arch_spec, max_tokens,
            )
        elif (
            self._cfg.sequential_generation
            and blueprint is not None
            and self._is_valid_blueprint(blueprint)
        ):
            files = self._phase2_sequential_generate(
                topic, exp_plan, _metric, _pkg, arch_spec, blueprint,
            )
            if self._cfg.hard_validation:
                files = self._hard_validate_and_repair(
                    files, topic, exp_plan, _metric, _pkg, arch_spec,
                )
            files = self._exec_fix_loop(files)
            best = SolutionNode(node_id="sequential", files=files, runs_ok=True, score=1.0)
        else:
            if self._cfg.sequential_generation and blueprint is None:
                self._log_event(
                    "Sequential generation requested but blueprint invalid — falling back"
                )
            files = self._phase2_generate_and_fix(
                topic, exp_plan, _metric, _pkg, arch_spec, max_tokens,
            )
            if self._cfg.hard_validation and files:
                files = self._hard_validate_and_repair(
                    files, topic, exp_plan, _metric, _pkg, arch_spec,
                )
            best = SolutionNode(
                node_id="single", files=files,
                runs_ok=bool(files), score=1.0 if files else 0.0,
            )

        review_rounds = 0
        if self._cfg.review_max_rounds > 0:
            best.files, review_rounds = self._phase4_review(
                best.files, topic, exp_plan, _metric,
            )

        elapsed = time.time() - t0
        self._log_event(
            f"CodeAgent.generate() done in {elapsed:.1f}s — "
            f"{self._calls} LLM calls, {self._runs} sandbox runs"
        )
        return CodeAgentResult(
            files=best.files,
            provider_name="advanced",
            elapsed_sec=elapsed,
            architecture_spec=arch_spec,
            validation_log=list(self._log),
            total_llm_calls=self._calls,
            total_sandbox_runs=self._runs,
            best_score=best.score,
            tree_nodes_explored=nodes_explored,
            review_rounds=review_rounds,
        )

    # -- CodeAgentProvider: refine ----------------------------------------- #

    def refine(
        self,
        *,
        current_files: dict[str, str],
        run_summaries: list[str],
        metric_key: str,
        metric_direction: str,
        topic: str,
        extra_hints: str,
        workdir: Path,
        timeout_sec: int = 600,
    ) -> CodeAgentResult:
        """Refine existing experiment code using the exec-fix loop."""
        t0 = time.time()
        self._log_event("CodeAgent.refine() started")
        _seed_workdir(workdir, current_files)

        # Build refinement context and pass it to _fix_runtime_error internals
        summaries_str = "\n".join(run_summaries[:20])
        files_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in sorted(current_files.items())
        )
        refine_prompt = (
            f"Improve this experiment code for topic: {topic}\n"
            f"Target: {metric_direction} '{metric_key}'\n"
            f"Prior run summaries:\n{summaries_str}\n{extra_hints}\n\n"
            f"Current code:\n{files_ctx}"
        )
        try:
            resp = self._chat(
                system=(
                    "You are an expert ML engineer. Improve the experiment code to "
                    "achieve better results on the specified metric. Return complete "
                    "files using ```filename:name.py ... ``` blocks."
                ),
                user=refine_prompt,
                max_tokens=8192,
            )
            files = _extract_files_from_text(resp)
            if not files:
                code = _extract_single_block(resp)
                if code.strip():
                    files = {"main.py": code}
            if self._cfg.hard_validation and files:
                files = self._hard_validate_and_repair(
                    files, topic, "", metric_key, "", "",
                )
            files = self._exec_fix_loop(files)
        except Exception as exc:
            logger.error("CodeAgent.refine failed: %s", exc)
            return CodeAgentResult(
                files={}, provider_name="advanced",
                elapsed_sec=time.time() - t0, error=str(exc),
                validation_log=list(self._log),
            )

        return CodeAgentResult(
            files=files,
            provider_name="advanced",
            elapsed_sec=time.time() - t0,
            validation_log=list(self._log),
            total_llm_calls=self._calls,
            total_sandbox_runs=self._runs,
        )

    # -- CodeAgentProvider: repair ----------------------------------------- #

    def repair(
        self,
        *,
        files: dict[str, str],
        issues: str,
        workdir: Path,
        timeout_sec: int = 300,
    ) -> CodeAgentResult:
        """Fix validation or runtime issues using the hard-validate pipeline."""
        t0 = time.time()
        self._log_event("CodeAgent.repair() started")
        try:
            repaired = self._repair_critical_issues(files, issues)
            if self._cfg.hard_validation:
                repaired = self._hard_validate_and_repair(
                    repaired, "", "", "", "", "",
                )
        except Exception as exc:
            logger.error("CodeAgent.repair failed: %s", exc)
            return CodeAgentResult(
                files={}, provider_name="advanced",
                elapsed_sec=time.time() - t0, error=str(exc),
            )
        return CodeAgentResult(
            files=repaired,
            provider_name="advanced",
            elapsed_sec=time.time() - t0,
            validation_log=list(self._log),
            total_llm_calls=self._calls,
        )

    # ======================================================================
    # Phase implementations  (preserved exactly from pipeline/code_agent.py)
    # ======================================================================

    def _phase1_blueprint(
        self, topic: str, exp_plan: str, metric: str,
    ) -> tuple[str, dict[str, Any] | None]:
        domain_ctx = self._build_domain_context()
        code_search_ctx = ""
        if self._code_search_result and self._code_search_result.patterns.has_content:
            pats = self._code_search_result.patterns
            code_search_ctx = (
                f"\n\n## Reference Implementations Found\n"
                f"API patterns: {', '.join(pats.api_patterns[:5])}\n"
                f"Data patterns: {', '.join(pats.data_patterns[:3])}\n"
            )

        system = (
            "You are a research software architect. Produce a YAML implementation "
            "blueprint for experiment code. The blueprint MUST specify:\n"
            "- files: list of filenames in dependency order\n"
            "- For each file: purpose, classes, functions, tensor_shapes\n"
            "- entry_point: always 'main.py'\n"
            "- metric_output_format: how metrics are printed\n"
            "Respond ONLY with valid YAML inside a ```yaml block."
        )
        user = (
            f"Topic: {topic}\nMetric: {metric}\n\n"
            f"Experiment Plan:\n{exp_plan[:3000]}\n\n"
            f"{domain_ctx}{code_search_ctx}"
        )
        try:
            raw = self._chat(system, user, max_tokens=4096)
            yaml_match = re.search(r"```ya?ml\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
            yaml_text = yaml_match.group(1).strip() if yaml_match else raw
            blueprint = self._parse_blueprint(yaml_text)
            self._log_event(
                f"Phase 1 blueprint: {len(blueprint.get('files', [])) if blueprint else 0} files planned"
            )
            return yaml_text, blueprint
        except Exception as exc:
            self._log_event(f"Phase 1 blueprint failed: {exc}")
            return "", None

    def _build_domain_context(self) -> str:
        if self._domain_profile is None:
            return ""
        try:
            from research.domains.prompt_adapter import get_adapter
            adapter = get_adapter(self._domain_profile.domain_id)
            return adapter.code_context() if hasattr(adapter, "code_context") else ""
        except Exception:
            return f"Domain: {getattr(self._domain_profile, 'display_name', '')}\n"

    def _parse_blueprint(self, yaml_text: str) -> dict[str, Any] | None:
        try:
            import yaml
            data = yaml.safe_load(yaml_text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        try:
            import re as _bp_re
            files_match = _bp_re.search(r"files:\s*\n((?:\s+-\s+\S+\n?)+)", yaml_text)
            if files_match:
                fnames = _bp_re.findall(r"-\s+(\S+)", files_match.group(1))
                return {"files": fnames, "entry_point": "main.py"}
        except Exception:
            pass
        return None

    @staticmethod
    def _is_valid_blueprint(bp: dict[str, Any]) -> bool:
        files = bp.get("files", [])
        if not files or not isinstance(files, list):
            return False
        if len(files) < 1 or len(files) > 15:
            return False
        if "main.py" not in [
            (f if isinstance(f, str) else f.get("name", "")) for f in files
        ]:
            return False
        return True

    def _phase2_sequential_generate(
        self, topic: str, exp_plan: str, metric: str,
        pkg_hint: str, arch_spec: str, blueprint: dict[str, Any],
    ) -> dict[str, str]:
        files_spec = blueprint.get("files", [])
        generated: dict[str, str] = {}
        summaries: dict[str, str] = {}

        system = (
            "You are an expert ML engineer generating ONE file of a multi-file "
            "Python experiment. Follow the architecture specification exactly. "
            "Return ONLY the Python code for the requested file — no explanation."
        )
        for file_entry in files_spec:
            fname = file_entry if isinstance(file_entry, str) else file_entry.get("name", "")
            if not fname or not fname.endswith(".py"):
                continue
            summary_ctx = "\n".join(
                f"# {fn}:\n{sm}" for fn, sm in summaries.items()
            )
            user = (
                f"Topic: {topic}\nMetric: {metric}\n\n"
                f"Architecture:\n{arch_spec[:2000]}\n\n"
                f"Previously generated files (summaries):\n{summary_ctx}\n\n"
                f"Now generate: {fname}\n"
                f"Requirements: {json.dumps(file_entry) if isinstance(file_entry, dict) else ''}"
            )
            try:
                code = self._chat(system, user, max_tokens=6144)
                code = self._extract_single_file_code(code, fname)
                if code.strip():
                    generated[fname] = code
                    summaries[fname] = self._build_code_summary(fname, code)
                    self._log_event(f"Phase 2: generated {fname} ({len(code)} chars)")
            except Exception as exc:
                self._log_event(f"Phase 2: failed to generate {fname}: {exc}")

        if not generated:
            self._log_event("Phase 2: sequential generation produced no files — falling back")
            return self._phase2_generate_and_fix(topic, exp_plan, metric, pkg_hint, arch_spec, 8192)
        return generated

    @staticmethod
    def _extract_single_file_code(content: str, expected_name: str) -> str:
        # Try language-tagged code block first
        pattern = rf"```(?:python|py)?\s*(?:#{expected_name})?\s*\n(.*?)```"
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: if content looks like raw Python, use it
        stripped = content.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("#")
        ):
            return stripped
        # Strip any outer fences
        cleaned = re.sub(r"^```[a-z]*\s*", "", stripped, flags=re.IGNORECASE)
        return re.sub(r"\s*```$", "", cleaned).strip()

    def _build_code_summary(self, fname: str, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return f"# {fname}: (parse error)\n"
        lines = [f"# {fname}:"]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name for n in ast.walk(node)
                    if isinstance(n, ast.FunctionDef) and n.col_offset > 0
                ]
                lines.append(f"  class {node.name}: methods={methods[:5]}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.col_offset == 0:
                    args = [a.arg for a in node.args.args[:4]]
                    lines.append(f"  def {node.name}({', '.join(args)})")
        return "\n".join(lines[:20])

    def _hard_validate_and_repair(
        self, files: dict[str, str], topic: str, exp_plan: str,
        metric: str, pkg_hint: str, arch_spec: str,
    ) -> dict[str, str]:
        for attempt in range(self._cfg.hard_validation_max_repairs):
            issues = self._hard_validate(files)
            if not issues:
                self._log_event(f"Hard validation passed on attempt {attempt + 1}")
                return files
            self._log_event(
                f"Hard validation attempt {attempt + 1}: {len(issues)} issue(s)"
            )
            files = self._repair_critical_issues(files, "\n".join(issues))
        return files

    def _hard_validate(self, files: dict[str, str]) -> list[str]:
        issues: list[str] = []
        from research.experiment.validator import (
            CodeValidation, validate_code, format_issues_for_llm,
        )
        for fname, code in files.items():
            try:
                ast.parse(code)
            except SyntaxError as e:
                issues.append(f"{fname}: SyntaxError at line {e.lineno}: {e.msg}")
                continue
            result: CodeValidation = validate_code(code, filename=fname)
            if result.issues:
                issues.extend(
                    f"{fname}: {issue}" for issue in result.issues[:3]
                )
        if "main.py" not in files:
            issues.append("Missing required entry point: main.py")
        return issues

    def _repair_critical_issues(
        self, files: dict[str, str], issues: str,
    ) -> dict[str, str]:
        files_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        system = (
            "You are fixing Python experiment code. Fix ONLY the reported issues. "
            "Return ALL files (fixed and unchanged) using ```filename:name.py blocks."
        )
        user = f"Issues to fix:\n{issues}\n\nCurrent code:\n{files_ctx}"
        try:
            resp = self._chat(system, user, max_tokens=8192)
            repaired = _extract_files_from_text(resp)
            if repaired:
                # Keep original files that weren't touched
                return {**files, **repaired}
        except Exception as exc:
            self._log_event(f"Repair failed: {exc}")
        return files

    def _phase2_generate_and_fix(
        self, topic: str, exp_plan: str, metric: str,
        pkg_hint: str, arch_spec: str, max_tokens: int,
    ) -> dict[str, str]:
        files = self._generate_code(topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens)
        return self._exec_fix_loop(files)

    def _exec_fix_loop(self, files: dict[str, str]) -> dict[str, str]:
        if not self._sandbox_factory or not files:
            return files
        for i in range(self._cfg.exec_fix_max_iterations):
            result = self._run_in_sandbox(
                files, timeout_sec=self._cfg.exec_fix_timeout_sec,
            )
            if result is None or result.returncode == 0:
                break
            self._log_event(
                f"Exec-fix iteration {i + 1}: returncode={result.returncode}"
            )
            fixed = self._fix_runtime_error(files, result)
            if fixed:
                files = fixed
        return files

    def _generate_code(
        self, topic: str, exp_plan: str, metric: str,
        pkg_hint: str, arch_spec: str, max_tokens: int,
    ) -> dict[str, str]:
        system = (
            "You are an expert ML engineer. Generate complete, runnable Python "
            "experiment code. Return all files using ```filename:name.py blocks. "
            "The entry point MUST be main.py. Print metrics as 'name: value'."
        )
        user = (
            f"Topic: {topic}\nMetric: {metric}\n{pkg_hint}\n\n"
            f"Experiment Plan:\n{exp_plan[:3000]}\n\n"
        )
        if arch_spec:
            user += f"## ARCHITECTURE SPECIFICATION (follow this):\n{arch_spec[:2000]}\n\n"
        user += (
            "FORBIDDEN: subprocess, os.system, eval, exec, shutil, socket, network.\n"
            "Use deterministic seeds. Write all files to cwd."
        )
        raw = self._chat(system, user, max_tokens=max_tokens)
        files = _extract_files_from_text(raw)
        if not files:
            code = _extract_single_block(raw)
            if code.strip():
                files = {"main.py": code}
        self._log_event(f"Single-shot generation: {len(files)} files")
        return files

    def _fix_runtime_error(
        self, files: dict[str, str], result: Any,
    ) -> dict[str, str] | None:
        error_info = f"Return code: {result.returncode}\n"
        if result.stderr:
            error_info += f"Stderr:\n{result.stderr[-2000:]}\n"
        files_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        system = (
            "Fix the Python experiment code runtime error. Return ALL fixed files "
            "using ```filename:name.py blocks."
        )
        user = f"Error:\n{error_info}\n\nCode:\n{files_ctx}"
        try:
            resp = self._chat(system, user, max_tokens=8192)
            fixed = _extract_files_from_text(resp)
            return fixed if fixed else None
        except Exception as exc:
            self._log_event(f"Fix runtime error failed: {exc}")
            return None

    def _phase3_tree_search(
        self, topic: str, exp_plan: str, metric: str,
        pkg_hint: str, arch_spec: str, max_tokens: int,
    ) -> tuple[SolutionNode, int]:
        root_files = self._generate_code(topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens)
        root = SolutionNode(node_id="root", files=root_files)
        self._evaluate_node(root, metric)

        nodes: list[SolutionNode] = [root]
        best = root
        explored = 1

        for depth in range(self._cfg.tree_search_max_depth):
            new_nodes: list[SolutionNode] = []
            for parent in nodes[:self._cfg.tree_search_candidates]:
                for _ in range(self._cfg.tree_search_candidates):
                    variant_files = self._generate_code(
                        topic, exp_plan, metric, pkg_hint, arch_spec, max_tokens,
                    )
                    node = SolutionNode(
                        node_id=f"d{depth}_v{explored}",
                        files=variant_files,
                        parent_id=parent.node_id,
                        depth=depth + 1,
                        generation_method="tree_variant",
                    )
                    self._evaluate_node(node, metric)
                    explored += 1
                    new_nodes.append(node)
                    if node.score > best.score:
                        best = node
            nodes = sorted(new_nodes, key=lambda n: -n.score)

        self._log_event(
            f"Tree search: explored={explored}, best_score={best.score:.3f}"
        )
        return best, explored

    def _evaluate_node(self, node: SolutionNode, metric_key: str) -> None:
        result = self._run_in_sandbox(node.files, timeout_sec=self._cfg.tree_search_eval_timeout_sec)
        if result is None:
            return
        node.runs_ok = result.returncode == 0
        node.returncode = result.returncode
        node.stdout = result.stdout[:1000]
        node.stderr = result.stderr[:500]
        node.metrics = dict(result.metrics) if result.metrics else {}
        node.evaluated = True
        node.score = self._score_node(node, metric_key)

    @staticmethod
    def _score_node(node: SolutionNode, metric_key: str) -> float:
        if not node.runs_ok:
            return 0.0
        if metric_key in node.metrics:
            try:
                return float(node.metrics[metric_key])
            except (ValueError, TypeError):
                pass
        return 0.5 if node.runs_ok else 0.0

    def _phase4_review(
        self, files: dict[str, str], topic: str, exp_plan: str, metric: str,
    ) -> tuple[dict[str, str], int]:
        rounds = 0
        files_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        for _ in range(self._cfg.review_max_rounds):
            review = self._chat(
                system=(
                    "You are a senior ML researcher reviewing experiment code. "
                    "List up to 5 concrete improvements needed. Be specific."
                ),
                user=(
                    f"Topic: {topic}\nMetric: {metric}\n\n"
                    f"Code to review:\n{files_ctx[:6000]}"
                ),
                max_tokens=1500,
            )
            if any(word in review.lower() for word in ("lgtm", "no issues", "looks good")):
                break
            fixed = self._chat(
                system=(
                    "You are an expert ML engineer. Apply the reviewer's feedback "
                    "to improve the code. Return ALL files using ```filename:name.py blocks."
                ),
                user=f"Reviewer feedback:\n{review}\n\nCode:\n{files_ctx[:6000]}",
                max_tokens=8192,
            )
            new_files = _extract_files_from_text(fixed)
            if new_files:
                files = new_files
                files_ctx = "\n\n".join(
                    f"```filename:{f}\n{c}\n```" for f, c in files.items()
                )
            rounds += 1
        return files, rounds

    def _chat(self, system: str, user: str, max_tokens: int = 8192) -> str:
        from research.pipeline._helpers import _chat_with_prompt
        resp = _chat_with_prompt(self._llm, system, user, max_tokens=max_tokens)
        self._calls += 1
        return resp.content if hasattr(resp, "content") else str(resp)

    def _get_or_create_sandbox(self) -> _SandboxLike:
        if self._sandbox is None and self._sandbox_factory and self._exp_config:
            self._sandbox = self._sandbox_factory(self._exp_config, self._stage_dir)
        return self._sandbox  # type: ignore[return-value]

    def _run_in_sandbox(
        self, files: dict[str, str], timeout_sec: int = 300,
    ) -> Any | None:
        sandbox = self._get_or_create_sandbox()
        if sandbox is None:
            return None
        import tempfile
        with tempfile.TemporaryDirectory(prefix="ca_run_") as tmp:
            tmp_path = Path(tmp)
            for fname, code in files.items():
                (tmp_path / fname).write_text(code, encoding="utf-8")
            try:
                result = sandbox.run_project(
                    tmp_path, entry_point="main.py", timeout_sec=timeout_sec,
                )
                self._runs += 1
                return result
            except Exception as exc:
                self._log_event(f"Sandbox run failed: {exc}")
                return None

    def _log_event(self, msg: str) -> None:
        self._log.append(msg)
        logger.debug("[CodeAgent] %s", msg)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_code_agent(
    config: RCConfig,
    llm: Any | None = None,
    prompts: Any | None = None,
    stage_dir: Path | None = None,
    sandbox_factory: Any | None = None,
    domain_profile: Any | None = None,
    code_search_result: Any | None = None,
) -> CodeAgentProvider:
    """Create the appropriate code agent backend from config.

    Provider selection via ``config.experiment.cli_agent.provider``:

    ``"llm"``        — LlmCodeAgent (LLM chat, backward-compatible default)
    ``"advanced"``   — CodeAgent (multi-phase: blueprint → tree search → review)
    ``"claude_code"``— ClaudeCodeAgent (Claude Code CLI)
    ``"codex"``      — CodexAgent (OpenAI Codex CLI)
    """
    agent_cfg = config.experiment.cli_agent
    provider = agent_cfg.provider

    if provider == "llm":
        if llm is None:
            raise RuntimeError("LLM code agent requires an LLM client")
        from research.prompts import PromptManager
        return LlmCodeAgent(llm, prompts or PromptManager(), config)  # type: ignore[return-value]

    if provider == "advanced":
        if llm is None:
            raise RuntimeError("Advanced code agent requires an LLM client")
        from research.prompts import PromptManager
        ca_cfg = agent_cfg if isinstance(agent_cfg, CodeAgentConfig) else CodeAgentConfig()
        return CodeAgent(  # type: ignore[return-value]
            llm=llm,
            prompts=prompts or PromptManager(),
            config=ca_cfg,
            stage_dir=stage_dir or Path("."),
            sandbox_factory=sandbox_factory,
            experiment_config=config.experiment,
            domain_profile=domain_profile,
            code_search_result=code_search_result,
        )

    if provider == "claude_code":
        binary = agent_cfg.binary_path or shutil.which("claude")
        if not binary:
            raise RuntimeError(
                "Claude Code binary not found. "
                "Install it or set experiment.code_agent.binary_path."
            )
        return ClaudeCodeAgent(  # type: ignore[return-value]
            binary_path=binary,
            model=agent_cfg.model or "sonnet",
            max_budget_usd=agent_cfg.max_budget_usd,
            timeout_sec=agent_cfg.timeout_sec,
            extra_args=list(agent_cfg.extra_args),
        )

    if provider == "codex":
        binary = agent_cfg.binary_path or shutil.which("codex")
        if not binary:
            raise RuntimeError(
                "Codex binary not found. "
                "Install it or set experiment.code_agent.binary_path."
            )
        return CodexAgent(  # type: ignore[return-value]
            binary_path=binary,
            model=agent_cfg.model or "",
            max_budget_usd=agent_cfg.max_budget_usd,
            timeout_sec=agent_cfg.timeout_sec,
            extra_args=list(agent_cfg.extra_args),
        )

    raise ValueError(
        f"Unknown code agent provider: {provider!r}. "
        "Choose from: 'llm', 'advanced', 'claude_code', 'codex'."
    )
