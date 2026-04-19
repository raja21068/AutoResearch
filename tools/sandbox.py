"""
core/tools/sandbox.py — Isolated code execution.

Uses Docker if available, otherwise subprocess fallback.
Supports: run code, run tests, auto-debug loop.
"""

import io
import logging
import os
import subprocess
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


class DockerSandbox:
    def __init__(self) -> None:
        self.image   = os.getenv("SANDBOX_IMAGE", "python:3.11-slim")
        self.timeout = int(os.getenv("SANDBOX_TIMEOUT", "60"))
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
                ["python3", filepath],
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
        with tempfile.TemporaryDirectory(prefix="nexus_") as tmpdir:
            code_path = os.path.join(tmpdir, "solution.py")
            with open(code_path, "w") as f:
                f.write(code)

            try:
                result = subprocess.run(
                    ["python3", code_path],
                    capture_output=True, text=True,
                    timeout=self.timeout, cwd=tmpdir,
                )
                stdout, stderr = result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return "", f"Timed out after {self.timeout}s"
            except Exception as e:
                return "", str(e)

            if test_code:
                # Prepend the code into the test file so tests are self-contained
                # (avoids import issues with Python built-in module names)
                combined_test = (
                    "# === Code under test ===\n"
                    f"{code}\n\n"
                    "# === Tests ===\n"
                    f"{test_code}\n"
                )
                test_path = os.path.join(tmpdir, "test_solution.py")
                with open(test_path, "w") as f:
                    f.write(combined_test)
                try:
                    tr = subprocess.run(
                        ["python3", "-m", "pytest", test_path, "-v", "--tb=short"],
                        capture_output=True, text=True,
                        timeout=self.timeout, cwd=tmpdir,
                    )
                    stdout += "\n--- pytest ---\n" + tr.stdout
                    if tr.stderr:
                        stderr += "\n" + tr.stderr
                except Exception as e:
                    stderr += f"\nTest error: {e}"

            return stdout, stderr
