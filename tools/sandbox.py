"""
core/tools/sandbox.py — Isolated code execution.

Uses Docker if available, otherwise subprocess fallback.
Supports: run code, run tests, auto-debug loop.
"""

import io
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Tuple

try:
    import docker
    import docker.errors
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

logger = logging.getLogger(__name__)

# ── Packages that are safe to import in the combined test file ─────────────
_SAFE_TEST_PACKAGES = {
    "torch", "numpy", "np", "pandas", "pd", "sklearn", "scipy",
    "matplotlib", "pytest", "json", "os", "sys", "re", "time",
    "random", "math", "collections", "typing", "unittest", "mock",
    "transformers", "datasets", "evaluate", "tqdm", "io",
    "pathlib", "tempfile", "shutil", "functools", "itertools",
    "abc", "dataclasses", "copy", "warnings", "logging",
}

# ── Unicode replacements for non-ASCII chars that break Python files on Windows
_UNICODE_REPLACEMENTS = {
    "\u00b1": "+/-",   # ± (0xb1)  — most common offender
    "\u2013": "-",     # –  en-dash
    "\u2014": "--",    # —  em-dash
    "\u2019": "'",     # '  right single quotation
    "\u2018": "'",     # '  left single quotation
    "\u201c": '"',     # "  left double quotation
    "\u201d": '"',     # "  right double quotation
    "\u00d7": "x",     # ×  multiplication sign
    "\u00e9": "e",     # é
    "\u00e8": "e",     # è
    "\u00e0": "a",     # à
    "\u03b1": "alpha", # α
    "\u03b2": "beta",  # β
    "\u03bb": "lambda",# λ
    "\u221e": "inf",   # ∞
    "\u2264": "<=",    # ≤
    "\u2265": ">=",    # ≥
}


def _sanitize_encoding(text: str) -> str:
    """Replace problematic Unicode characters with safe ASCII equivalents.

    On Windows, Python subprocess files must be valid UTF-8 or declare an
    encoding.  When LLMs emit characters like ± (U+00B1) and the string is
    written without a BOM/encoding declaration, the Python parser raises:
      SyntaxError: Non-UTF-8 code starting with '\\xb1' in file ...
    We replace the worst offenders with ASCII equivalents and prepend an
    encoding declaration so the file parses cleanly on any platform.
    """
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Encode to UTF-8 safely — replace any remaining non-UTF-8 bytes
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return text


def _is_latex(code: str) -> bool:
    """Return True if the content looks like LaTeX rather than Python."""
    stripped = code.strip()
    return (
        stripped.startswith("\\documentclass")
        or stripped.startswith("%!TEX")
        or (stripped.startswith("\\") and "\\begin{document}" in stripped[:500])
    )


def _sanitize_test_imports(test_code: str) -> str:
    """Strip top-level imports of unknown modules from generated test code.

    When the sandbox prepends solution code to the test file, all classes and
    functions are already in scope.  Any ``from preprocessing import X`` or
    ``from your_module import X`` lines will raise ModuleNotFoundError.
    This function comments them out as a second line of defence (the first
    defence is TesterAgent._strip_module_imports).
    """
    import re
    kept = []
    for line in test_code.splitlines():
        stripped = line.strip()
        m_from   = re.match(r'from\s+(\w+)', stripped)
        m_import = re.match(r'^import\s+(\w+)', stripped)
        module   = (m_from or m_import)
        if module:
            pkg = module.group(1)
            if pkg not in _SAFE_TEST_PACKAGES:
                kept.append(f"# REMOVED (module not available in sandbox): {stripped}")
                continue
        kept.append(line)
    return "\n".join(kept)


def _get_python_executable() -> str:
    """Resolve the Python executable for subprocess execution.

    Priority: CONDA_PYTHON_PATH → SANDBOX_PYTHON → sys.executable
    This fixes [WinError 2] where 'python3' does not exist on Windows.
    """
    for env_var in ("CONDA_PYTHON_PATH", "SANDBOX_PYTHON"):
        val = os.getenv(env_var, "").strip()
        if val:
            return val
    return sys.executable


class DockerSandbox:
    def __init__(self) -> None:
        self.image   = os.getenv("SANDBOX_IMAGE", "python:3.11-slim")
        self.timeout = int(os.getenv("SANDBOX_TIMEOUT", "300"))
        self.client  = None
        if HAS_DOCKER:
            try:
                self.client = docker.from_env()
                self.client.ping()
                logger.info("Docker sandbox ready (image=%s)", self.image)
            except Exception:
                self.client = None
                logger.info("Docker not available — using subprocess fallback")

    # ── Public API ────────────────────────────────────────────

    def run_code(self, code: str, test_code: str = "") -> Tuple[str, str]:
        """Execute code (and optional tests). Returns (stdout, stderr)."""
        if self.client:
            return self._run_docker(code, test_code)
        return self._run_subprocess(code, test_code)

    def run_file(self, filepath: str) -> Tuple[str, str]:
        """Execute a Python file. Returns (stdout, stderr)."""
        try:
            result = subprocess.run(
                [_get_python_executable(), filepath],
                capture_output=True, text=True, timeout=self.timeout,
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return "", f"Timed out after {self.timeout}s"
        except Exception as e:
            return "", str(e)

    def run_command(self, cmd: str, cwd: str = ".") -> Tuple[str, str]:
        """Execute a shell command. Returns (stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=self.timeout, cwd=cwd,
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return "", f"Timed out after {self.timeout}s"
        except Exception as e:
            return "", str(e)

    # ── Docker backend ────────────────────────────────────────

    def _run_docker(self, code: str, test_code: str) -> Tuple[str, str]:
        container = None
        try:
            container = self.client.containers.run(
                self.image, command="sleep 120",
                detach=True, remove=False,
                mem_limit="1g", nano_cpus=1_000_000_000,
                pids_limit=128, network_disabled=True,
                tmpfs={"/workspace": "size=256m,exec"},
            )
            self._inject(container, "code.py", code)
            _, out = container.exec_run(
                "bash -c 'cd /workspace && python /tmp/code.py'", demux=False,
            )
            stdout = out.decode("utf-8", errors="replace") if out else ""
            stderr = ""

            if test_code:
                self._inject(container, "test_code.py", test_code)
                container.exec_run("pip install pytest -q")
                _, tout = container.exec_run(
                    "bash -c 'cd /tmp && python -m pytest test_code.py -v --tb=short'",
                    demux=False,
                )
                stdout += "\n--- pytest ---\n" + (tout.decode("utf-8", errors="replace") if tout else "")
            return stdout, stderr
        except Exception as e:
            return "", f"Docker error: {e}"
        finally:
            if container:
                try: container.remove(force=True)
                except Exception: pass

    def _inject(self, container, filename, content, dest="/tmp"):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            encoded = content.encode("utf-8")
            info = tarfile.TarInfo(name=filename)
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
        container.put_archive(dest, buf.getvalue())

    # ── Subprocess backend ────────────────────────────────────

    def _run_subprocess(self, code: str, test_code: str) -> Tuple[str, str]:
        # ── Guard: skip Python execution for LaTeX documents ──────────────────
        if _is_latex(code):
            logger.info("Sandbox: detected LaTeX content — skipping Python execution")
            return (
                "LaTeX document detected — no Python execution needed.\n"
                "Paper saved for compilation.\n",
                ""  # no stderr
            )

        # ── Sanitize encoding (± and other non-ASCII → ASCII) ─────────────────
        code = _sanitize_encoding(code)
        # Prepend UTF-8 encoding declaration so Python accepts any remaining
        # characters without SyntaxError on Windows.
        if not code.startswith("# -*- coding"):
            code = "# -*- coding: utf-8 -*-\n" + code

        with tempfile.TemporaryDirectory(prefix="nexus_") as tmpdir:
            code_path = os.path.join(tmpdir, "solution.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            try:
                result = subprocess.run(
                    [_get_python_executable(), code_path],
                    capture_output=True, text=True,
                    timeout=self.timeout, cwd=tmpdir,
                    encoding="utf-8", errors="replace",
                )
                stdout, stderr = result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return "", f"Timed out after {self.timeout}s"
            except Exception as e:
                return "", str(e)

            if test_code:
                # Sanitize test imports before combining:
                # The combined file already has all solution code in scope so any
                # "from preprocessing import X" or "from your_module import X" will
                # raise ModuleNotFoundError.  Strip those lines defensively.
                test_code = _sanitize_test_imports(test_code)
                # Also sanitize encoding in tests
                test_code = _sanitize_encoding(test_code)

                # Prepend the code into the test file so tests are self-contained
                # (avoids import issues with Python built-in module names)
                combined_test = (
                    "# -*- coding: utf-8 -*-\n"
                    "# === Code under test ===\n"
                    f"{code}\n\n"
                    "# === Tests ===\n"
                    f"{test_code}\n"
                )
                test_path = os.path.join(tmpdir, "test_solution.py")
                with open(test_path, "w", encoding="utf-8") as f:
                    f.write(combined_test)
                try:
                    tr = subprocess.run(
                        [_get_python_executable(), "-m", "pytest", test_path, "-v", "--tb=short"],
                        capture_output=True, text=True,
                        timeout=self.timeout, cwd=tmpdir,
                        encoding="utf-8", errors="replace",
                    )
                    stdout += "\n--- pytest ---\n" + tr.stdout
                    if tr.stderr:
                        stderr += "\n" + tr.stderr
                except Exception as e:
                    stderr += f"\nTest error: {e}"

            return stdout, stderr