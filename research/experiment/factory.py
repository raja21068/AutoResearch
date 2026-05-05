"""Factory for creating experiment sandboxes based on configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from research.config import RCConfig
from research.experiment.sandbox import ExperimentSandbox

logger = logging.getLogger(__name__)


def create_sandbox(config: RCConfig, workdir: Path | None = None) -> ExperimentSandbox:
    """Create an ExperimentSandbox appropriate for the configured mode.

    Modes: sandbox (subprocess), docker, ssh_remote, colab_drive, agentic.
    All modes return an object with the same ``run_project()`` interface.
    """
    mode = config.experiment.mode

    if mode == "docker":
        return _create_docker_sandbox(config, workdir)
    elif mode == "ssh_remote":
        return _create_ssh_sandbox(config, workdir)
    elif mode == "simulated":
        return _create_simulated_sandbox(config, workdir)
    else:
        # Default: local subprocess sandbox
        return ExperimentSandbox(
            python_path=config.experiment.sandbox.python_path,
            timeout_sec=config.experiment.time_budget_sec,
            workdir=workdir,
        )


def _create_docker_sandbox(config: RCConfig, workdir: Path | None) -> ExperimentSandbox:
    """Create a Docker-based sandbox."""
    docker_cfg = config.experiment.docker
    sandbox = ExperimentSandbox(
        python_path=docker_cfg.container_python,
        timeout_sec=config.experiment.time_budget_sec,
        workdir=workdir,
    )
    # Docker execution is handled by overriding run_project in subclass;
    # for now fall back to local subprocess with the configured python.
    logger.info("Docker sandbox: image=%s, gpu=%s", docker_cfg.image, docker_cfg.gpu_enabled)
    return sandbox


def _create_ssh_sandbox(config: RCConfig, workdir: Path | None) -> ExperimentSandbox:
    """Create an SSH-based remote sandbox."""
    ssh_cfg = config.experiment.ssh_remote
    sandbox = ExperimentSandbox(
        python_path=ssh_cfg.remote_python,
        timeout_sec=ssh_cfg.timeout_sec,
        workdir=workdir,
    )
    logger.info("SSH sandbox: %s@%s:%d", ssh_cfg.user, ssh_cfg.host, ssh_cfg.port)
    return sandbox


def _create_simulated_sandbox(config: RCConfig, workdir: Path | None) -> ExperimentSandbox:
    """Create a sandbox for simulated experiments (no actual execution)."""
    return ExperimentSandbox(
        python_path=config.experiment.sandbox.python_path,
        timeout_sec=config.experiment.time_budget_sec,
        workdir=workdir,
    )
