"""Hardware profiling — detect GPU / MPS / CPU capabilities.

Used by the research pipeline to:
* Choose appropriate compute budget strings for experiment prompts
* Decide whether to install PyTorch GPU or CPU builds
* Filter metric names vs. log noise in experiment output
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    """Snapshot of the machine's compute capabilities."""

    gpu_type: str = "cpu"           # "cuda" | "mps" | "cpu"
    gpu_name: str = "none"
    gpu_count: int = 0
    vram_mb: int = 0
    has_gpu: bool = False
    cpu_cores: int = 0
    ram_mb: int = 0
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "vram_mb": self.vram_mb,
            "has_gpu": self.has_gpu,
            "cpu_cores": self.cpu_cores,
            "ram_mb": self.ram_mb,
            "warning": self.warning,
        }

    @property
    def compute_budget_hint(self) -> str:
        if self.gpu_type == "cuda" and self.vram_mb >= 16000:
            return "large GPU available — can train medium models"
        if self.gpu_type == "cuda":
            return "GPU available — keep model size small"
        if self.gpu_type == "mps":
            return "Apple MPS available — use MPS device, keep models small"
        return "CPU only — use lightweight experiments, avoid heavy training"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_hardware(*, ssh_config: Any | None = None) -> HardwareProfile:
    """Detect local (or remote via SSH) hardware capabilities."""
    if ssh_config and getattr(ssh_config, "host", ""):
        return _detect_remote(ssh_config)
    return _detect_local()


def _detect_local() -> HardwareProfile:
    profile = HardwareProfile()

    # CPU cores
    try:
        profile.cpu_cores = os.cpu_count() or 1
    except Exception:
        profile.cpu_cores = 1

    # RAM
    try:
        import platform
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                          text=True, timeout=5).strip()
            profile.ram_mb = int(out) // (1024 * 1024)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        profile.ram_mb = int(line.split()[1]) // 1024
                        break
    except Exception:
        pass

    # NVIDIA GPU via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,count",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=10,
            ).strip()
            if out:
                parts = out.split("\n")[0].split(",")
                profile.gpu_name = parts[0].strip()
                profile.vram_mb = int(float(parts[1].strip()))
                profile.gpu_count = int(parts[2].strip()) if len(parts) > 2 else 1
                profile.gpu_type = "cuda"
                profile.has_gpu = True
                return profile
        except Exception as exc:
            logger.debug("nvidia-smi failed: %s", exc)

    # Apple MPS
    try:
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            profile.gpu_type = "mps"
            profile.gpu_name = "Apple Silicon"
            profile.has_gpu = True
            # Approximate VRAM as unified memory
            profile.vram_mb = profile.ram_mb
            return profile
    except Exception:
        pass

    profile.warning = "No GPU detected — experiments will run on CPU only"
    return profile


def _detect_remote(ssh_config: Any) -> HardwareProfile:
    """Detect hardware on a remote host via SSH."""
    host = getattr(ssh_config, "host", "")
    user = getattr(ssh_config, "user", "")
    port = getattr(ssh_config, "port", 22)
    key_path = getattr(ssh_config, "key_path", "")

    if not host:
        return HardwareProfile(warning="No SSH host configured")

    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port)]
    if key_path:
        ssh_cmd.extend(["-i", key_path])
    ssh_cmd.append(f"{user}@{host}" if user else host)

    profile = HardwareProfile()
    try:
        out = subprocess.check_output(
            ssh_cmd + ["nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo CPU_ONLY"],
            text=True, timeout=15,
        ).strip()
        if "CPU_ONLY" not in out and out:
            parts = out.split("\n")[0].split(",")
            profile.gpu_name = parts[0].strip()
            profile.vram_mb = int(float(parts[1].strip()))
            profile.gpu_type = "cuda"
            profile.has_gpu = True
        else:
            profile.warning = f"Remote host {host}: No GPU detected"
    except Exception as exc:
        profile.warning = f"SSH hardware detection failed: {exc}"

    return profile


# ---------------------------------------------------------------------------
# PyTorch availability check
# ---------------------------------------------------------------------------


def ensure_torch_available(python_path: str, gpu_type: str = "cpu") -> bool:
    """Check if PyTorch is importable via the given Python interpreter."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Metric name detection (used by experiment output parsers)
# ---------------------------------------------------------------------------

_METRIC_KEYWORDS = {
    "accuracy", "loss", "f1", "precision", "recall", "auc", "roc",
    "mae", "mse", "rmse", "r2", "bleu", "rouge", "perplexity",
    "reward", "regret", "score", "error", "rate", "metric",
    "train", "val", "test", "eval", "epoch", "step", "iter",
}

_LOG_NOISE = {
    "downloading", "loading", "saving", "checkpoint", "warning",
    "info", "debug", "cuda", "device", "using", "running",
    "starting", "finished", "processing", "elapsed", "eta",
}


def is_metric_name(name: str) -> bool:
    """Return True if *name* looks like a metric rather than a log/status line."""
    lower = name.lower().strip()
    if not lower or len(lower) > 80:
        return False
    tokens = set(re.split(r"[\s_\-/]+", lower))
    if tokens & _METRIC_KEYWORDS:
        return True
    if tokens & _LOG_NOISE and not (tokens & _METRIC_KEYWORDS):
        return False
    # If it contains numbers or looks like key=value, likely a metric
    if re.match(r"^[\w\s_\-./]+$", lower):
        return True
    return False
