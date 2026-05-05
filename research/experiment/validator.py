"""Code validation for generated experiment scripts.

Performs AST-based structural checks and common-issue detection
before code is executed in the sandbox.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class CodeValidation:
    """Result of validating a single code file."""

    filename: str
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_main_guard: bool = False
    imports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.valid and not self.errors


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate_code(
    code: str,
    filename: str = "main.py",
    *,
    allowed_imports: set[str] | None = None,
) -> CodeValidation:
    """Validate Python source code via AST parsing and heuristic checks.

    Returns a ``CodeValidation`` with errors, warnings, and metadata.
    """
    result = CodeValidation(filename=filename)

    if not code or not code.strip():
        result.valid = False
        result.errors.append("Empty source code")
        return result

    # -- AST parse ---------------------------------------------------------
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.valid = False
        result.errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")
        return result

    # -- Walk AST for metadata ---------------------------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result.imports.append(node.module)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            result.functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            result.classes.append(node.name)

    # -- Check for if __name__ == "__main__" guard -------------------------
    result.has_main_guard = _has_main_guard(code)

    # -- Dangerous patterns ------------------------------------------------
    _dangerous = [
        (r"\bos\.system\b", "os.system() is unsafe — use subprocess instead"),
        (r"\beval\s*\(", "eval() is a security risk"),
        (r"\bexec\s*\(", "exec() is a security risk"),
        (r"\b__import__\b", "__import__() is discouraged"),
    ]
    for pattern, msg in _dangerous:
        if re.search(pattern, code):
            result.warnings.append(msg)

    # -- Import allowlist check --------------------------------------------
    if allowed_imports is not None:
        for imp in result.imports:
            root = imp.split(".")[0]
            if root not in allowed_imports and root not in _STDLIB_MODULES:
                result.warnings.append(f"Import '{imp}' not in allowed list")

    # -- Common issues -----------------------------------------------------
    if "plt.show()" in code:
        result.warnings.append("plt.show() will block in headless mode — use plt.savefig()")

    if not result.has_main_guard and filename == "main.py":
        result.warnings.append("main.py missing 'if __name__ == \"__main__\"' guard")

    return result


def _has_main_guard(code: str) -> bool:
    return bool(re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', code))


# ---------------------------------------------------------------------------
# Format issues for LLM feedback
# ---------------------------------------------------------------------------


def format_issues_for_llm(validations: list[CodeValidation]) -> str:
    """Format validation results into a prompt-friendly string for the LLM."""
    if not validations:
        return ""

    parts: list[str] = []
    for v in validations:
        if v.errors or v.warnings:
            parts.append(f"## {v.filename}")
            for e in v.errors:
                parts.append(f"  ERROR: {e}")
            for w in v.warnings:
                parts.append(f"  WARNING: {w}")

    if not parts:
        return ""

    return "=== CODE VALIDATION ISSUES ===\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# Standard library module names (subset for import checking)
# ---------------------------------------------------------------------------

_STDLIB_MODULES = {
    "abc", "argparse", "ast", "asyncio", "base64", "bisect",
    "collections", "contextlib", "copy", "csv", "dataclasses",
    "datetime", "decimal", "enum", "functools", "glob", "hashlib",
    "heapq", "html", "http", "importlib", "inspect", "io",
    "itertools", "json", "logging", "math", "multiprocessing",
    "operator", "os", "pathlib", "pickle", "platform", "pprint",
    "queue", "random", "re", "shutil", "signal", "socket",
    "sqlite3", "statistics", "string", "struct", "subprocess",
    "sys", "tempfile", "textwrap", "threading", "time", "traceback",
    "typing", "unittest", "urllib", "uuid", "warnings", "xml", "yaml",
}
