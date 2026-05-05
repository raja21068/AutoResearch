"""Bridge: ``from research.experiment.code_agent import create_code_agent``

Delegates to the unified code agent in ``research.pipeline.code_agent``.
This module exists so that legacy imports resolve correctly.
"""

from __future__ import annotations

from typing import Any

from research.config import RCConfig


def create_code_agent(config: RCConfig, *, llm: Any = None, prompts: Any = None) -> Any:
    """Factory that returns the appropriate CodeAgentProvider.

    The heavy implementation lives in ``research.pipeline.code_agent``;
    this thin wrapper keeps the public import path stable.
    """
    from research.pipeline.code_agent import (
        LlmCodeAgent,
        CodeAgent,
    )

    cli_cfg = config.experiment.cli_agent
    provider = cli_cfg.provider.lower()

    if provider == "advanced" or (
        provider == "llm" and config.experiment.code_agent.enabled
    ):
        # Use advanced multi-phase code agent
        try:
            return CodeAgent(
                llm=llm,
                prompts=prompts,
                config=config,
            )
        except Exception:
            pass  # Fall through to basic LLM agent

    # Default: basic LLM chat agent
    return LlmCodeAgent(llm=llm, prompts=prompts, config=config)
