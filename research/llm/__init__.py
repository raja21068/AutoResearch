"""research.llm — Unified, provider-agnostic LLM client for the research pipeline.

All LLM calls in the research pipeline flow through this module.
Switching providers requires only .env / config.arc.yaml changes.
"""

from __future__ import annotations

from research.llm.client import LLMClient
from research.config import RCConfig

__all__ = ["LLMClient", "create_llm_client"]


def create_llm_client(config: RCConfig) -> LLMClient:
    """Factory: build an LLMClient from the resolved RCConfig."""
    llm_cfg = config.llm
    return LLMClient(
        provider=llm_cfg.provider,
        base_url=llm_cfg.base_url,
        api_key=llm_cfg.api_key or _resolve_api_key(llm_cfg.api_key_env),
        primary_model=llm_cfg.primary_model,
        fallback_models=list(llm_cfg.fallback_models),
        wire_api=llm_cfg.wire_api,
    )


def _resolve_api_key(env_var: str) -> str:
    """Read an API key from the environment variable named *env_var*."""
    import os

    if not env_var:
        return ""
    return os.getenv(env_var, "")
