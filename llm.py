"""
llm.py — Single source of truth for all LLM calls (agent layer).

Provides:
    llm()        — async, returns full response string
    llm_stream() — async generator, yields tokens as they arrive
    preflight()  — async connectivity test before long runs
    get_usage_stats() — token/cost/cache statistics

Fully LLM-agnostic: works with ANY provider via LiteLLM.
Switching providers requires ONLY .env changes — no code modifications.

Supported providers (via LiteLLM):
    OpenAI, Anthropic, Google Gemini, DeepSeek, Groq, Together AI,
    OpenRouter, Mistral, Cohere, Ollama, Azure OpenAI, AWS Bedrock,
    and 100+ more.

Configuration via environment variables:
    LLM_MODE          — litellm (default) | openai | deepseek | local
    LLM_PROVIDER      — legacy alias for LLM_MODE
    OPENAI_API_KEY    — for OpenAI / OpenAI-compatible providers
    DEEPSEEK_API_KEY  — for DeepSeek mode
    GEMINI_API_KEY    — for Google Gemini
    GROQ_API_KEY      — for Groq
    ANTHROPIC_API_KEY — for Anthropic
    LOCAL_LLM_URL     — for local mode (default: http://localhost:11434/v1)
    LLM_MODEL         — global model override (forces ALL agents to use this)
    LOCAL_MODEL       — model for local mode
    DEFAULT_MODEL     — fallback for unrecognised agents

Per-agent model routing (env overrides):
    PLANNER_MODEL       CODER_MODEL         TESTER_MODEL
    DEBUGGER_MODEL      CRITIC_MODEL        RESEARCHER_MODEL
    EXPERIMENT_MODEL    PAPER_WRITER_MODEL  FIGURE_MODEL

Tuning:
    LLM_TEMPERATURE    — temperature for all calls (default: 0.0)
    LLM_SEED           — seed for reproducibility (default: 42, 0 = disable)

Efficiency settings:
    LLM_CACHE_ENABLED  — 1/0 (default: 1) — deduplicate identical prompts
    LLM_MAX_RETRIES    — number of retries on transient failures (default: 3)
    LLM_RPM_LIMIT      — requests per minute rate limit (default: 0 = unlimited)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from threading import Lock
from typing import Any, AsyncGenerator

import litellm
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Silence LiteLLM's verbose success logs; keep warnings/errors.
litellm.set_verbose = False

# ── Mode Detection ─────────────────────────────────────────────────────────
LLM_MODE = os.getenv("LLM_MODE") or os.getenv("LLM_PROVIDER", "litellm")

# ── Unified Temperature & Seed ────────────────────────────────────────────
_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
_SEED_RAW = os.getenv("LLM_SEED", "42").strip()
_SEED: int | None = None if _SEED_RAW == "0" else int(_SEED_RAW)  # 0 = disable

# Initialize clients for non-LiteLLM modes
_openai_client: AsyncOpenAI | None = None
_deepseek_client: AsyncOpenAI | None = None
_local_client: AsyncOpenAI | None = None

if LLM_MODE == "openai":
    _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif LLM_MODE == "deepseek":
    _deepseek_client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
elif LLM_MODE == "local":
    _local_client = AsyncOpenAI(
        api_key="none",
        base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
    )

# ── Model routing ──────────────────────────────────────────────────────────

_ROUTING: dict[str, str] = {
    "planner":      "PLANNER_MODEL",
    "coder":        "CODER_MODEL",
    "tester":       "TESTER_MODEL",
    "debugger":     "DEBUGGER_MODEL",
    "critic":       "CRITIC_MODEL",
    "researcher":   "RESEARCHER_MODEL",
    "experiment":   "EXPERIMENT_MODEL",
    "paper_writer": "PAPER_WRITER_MODEL",
    "figure":       "FIGURE_MODEL",
}

_DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")

_DEFAULTS: dict[str, str] = {
    "planner":      os.getenv("PLANNER_MODEL", _DEFAULT_MODEL),
    "coder":        os.getenv("CODER_MODEL", _DEFAULT_MODEL),
    "tester":       os.getenv("TESTER_MODEL", _DEFAULT_MODEL),
    "debugger":     os.getenv("DEBUGGER_MODEL", _DEFAULT_MODEL),
    "critic":       os.getenv("CRITIC_MODEL", _DEFAULT_MODEL),
    "researcher":   os.getenv("RESEARCHER_MODEL", _DEFAULT_MODEL),
    "experiment":   os.getenv("EXPERIMENT_MODEL", _DEFAULT_MODEL),
    "paper_writer": os.getenv("PAPER_WRITER_MODEL", _DEFAULT_MODEL),
    "figure":       os.getenv("FIGURE_MODEL", _DEFAULT_MODEL),
}


def _resolve_model(agent: str) -> str:
    """Return the model string for *agent*, respecting env-var overrides.

    When LLM_MODE is 'deepseek' or 'openai' (direct client, not LiteLLM),
    strips the provider prefix (e.g. 'deepseek/deepseek-chat' → 'deepseek-chat')
    because direct OpenAI-compatible APIs expect bare model names.
    """
    global_override = os.getenv("LLM_MODEL")
    if global_override:
        model = global_override
    else:
        agent = agent.lower()

        env_key = _ROUTING.get(agent)
        model = ""
        if env_key:
            model = os.getenv(env_key, "")

        if not model and agent in _DEFAULTS:
            model = _DEFAULTS[agent]

        if not model:
            if LLM_MODE == "local":
                model = os.getenv("LOCAL_MODEL", "deepseek-coder")
            else:
                model = _DEFAULT_MODEL

    # Strip provider prefix for non-LiteLLM modes — direct OpenAI-compatible
    # clients expect bare model names (e.g. 'deepseek-chat' not 'deepseek/deepseek-chat')
    if LLM_MODE in ("deepseek", "openai", "local") and "/" in model:
        model = model.split("/", 1)[1]

    return model


def _get_client() -> AsyncOpenAI | None:
    """Return the appropriate client based on LLM_MODE."""
    if LLM_MODE == "openai":
        return _openai_client
    elif LLM_MODE == "deepseek":
        return _deepseek_client
    elif LLM_MODE == "local":
        return _local_client
    return None


# ── Response Cache ─────────────────────────────────────────────────────────

_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "1") == "1"
_cache: dict[str, str] = {}
_cache_lock = Lock()


def _cache_key(model: str, system: str, prompt: str) -> str:
    blob = f"{model}|{system}|{prompt}"
    return hashlib.sha256(blob.encode()).hexdigest()


# ── Retry settings ─────────────────────────────────────────────────────────

_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ── Rate limiter ───────────────────────────────────────────────────────────

_RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", "0"))
_rate_tokens = float(_RPM_LIMIT)
_rate_last = time.monotonic()
_rate_lock = Lock()


async def _rate_limit_acquire() -> None:
    """Async-friendly rate limiter."""
    global _rate_tokens, _rate_last
    if _RPM_LIMIT <= 0:
        return
    wait = 0.0
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _rate_last
        _rate_last = now
        _rate_tokens = min(_RPM_LIMIT, _rate_tokens + elapsed * (_RPM_LIMIT / 60.0))
        if _rate_tokens < 1:
            wait = (1 - _rate_tokens) / (_RPM_LIMIT / 60.0)
            _rate_tokens = 0
        else:
            _rate_tokens -= 1
    if wait > 0:
        await asyncio.sleep(wait)


# ── Usage tracking ─────────────────────────────────────────────────────────

_usage_stats: dict[str, float] = {
    "total_calls": 0,
    "cache_hits": 0,
    "retries": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "estimated_cost_usd": 0.0,
}
_usage_lock = Lock()


def get_usage_stats() -> dict[str, float]:
    """Return a copy of current usage statistics."""
    with _usage_lock:
        return dict(_usage_stats)


def _track_usage(response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage:
        with _usage_lock:
            _usage_stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            _usage_stats["completion_tokens"] += getattr(usage, "completion_tokens", 0)
    # Cost estimation via LiteLLM
    try:
        cost = litellm.completion_cost(completion_response=response)
        with _usage_lock:
            _usage_stats["estimated_cost_usd"] += cost
    except Exception:
        pass


# ── Think-tag stripping ───────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning-model responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Public API ─────────────────────────────────────────────────────────────

async def llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "",
    max_tokens: int | None = None,
) -> str:
    """Non-streaming call. Returns the full response string.

    Args:
        prompt: The user message.
        system: System prompt.
        agent:  Optional agent name for per-agent model routing.
                Supported: planner, coder, tester, debugger, critic,
                researcher, experiment, paper_writer, figure.
        max_tokens: Maximum output tokens. If None, uses per-agent defaults.
                    paper_writer/researcher default to 8192 to prevent truncation.
    """
    model = _resolve_model(agent)

    # Per-agent max_tokens defaults — paper agents need much higher limits
    # to produce complete IEEE sections without mid-sentence truncation.
    if max_tokens is None:
        _agent_defaults = {
            "paper_writer": 8192,
            "researcher":   8192,
            "coder":        4096,
            "debugger":     4096,
            "planner":      2048,
            "tester":       2048,
            "critic":       2048,
            "experiment":   4096,
        }
        max_tokens = _agent_defaults.get(agent.lower(), 4096)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]

    # ── Cache check ──
    ck = _cache_key(model, system, prompt)
    if _CACHE_ENABLED:
        with _cache_lock:
            if ck in _cache:
                with _usage_lock:
                    _usage_stats["cache_hits"] += 1
                return _cache[ck]

    # ── LiteLLM mode (default) ──
    if LLM_MODE == "litellm":
        result = await _litellm_call(model, messages, max_tokens=max_tokens)
    else:
        # ── OpenAI-compatible client ──
        client = _get_client()
        if client:
            result = await _openai_call(client, model, messages, max_tokens=max_tokens)
        else:
            logger.error("Invalid LLM_MODE: %s", LLM_MODE)
            return ""

    with _usage_lock:
        _usage_stats["total_calls"] += 1

    # ── Cache store (only non-empty results) ──
    if _CACHE_ENABLED and result:
        with _cache_lock:
            _cache[ck] = result

    return result


async def _litellm_call(model: str, messages: list[dict], max_tokens: int = 4096) -> str:
    """Call LiteLLM with retry and rate limiting."""
    last_exc: Exception | None = None
    for attempt in range(1 + _MAX_RETRIES):
        try:
            await _rate_limit_acquire()
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": _TEMPERATURE,
                "max_tokens": max_tokens,
            }
            if _SEED is not None:
                kwargs["seed"] = _SEED
            response = await litellm.acompletion(**kwargs)
            _track_usage(response)
            text = response.choices[0].message.content or ""
            return _strip_thinking(text)
        except Exception as exc:
            last_exc = exc
            with _usage_lock:
                _usage_stats["retries"] += 1
            if attempt < _MAX_RETRIES:
                delay = min(2 ** (attempt + 1), 30)
                logger.warning("LLM call failed (attempt %d/%d, model=%s): %s — retrying in %ds",
                               attempt + 1, 1 + _MAX_RETRIES, model, exc, delay)
                await asyncio.sleep(delay)
    logger.error("All retries exhausted for model %s: %s", model, last_exc)
    return ""


async def _openai_call(client: AsyncOpenAI, model: str, messages: list[dict], max_tokens: int = 4096) -> str:
    """Call OpenAI-compatible API with retry."""
    last_exc: Exception | None = None
    for attempt in range(1 + _MAX_RETRIES):
        try:
            await _rate_limit_acquire()
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": _TEMPERATURE,
                "max_tokens": max_tokens,
            }
            if _SEED is not None:
                kwargs["seed"] = _SEED
            response = await client.chat.completions.create(**kwargs)
            _track_usage(response)
            text = response.choices[0].message.content or ""
            return _strip_thinking(text)
        except Exception as exc:
            last_exc = exc
            with _usage_lock:
                _usage_stats["retries"] += 1
            if attempt < _MAX_RETRIES:
                delay = min(2 ** (attempt + 1), 30)
                logger.warning("LLM call failed (attempt %d/%d, mode=%s): %s — retrying in %ds",
                               attempt + 1, 1 + _MAX_RETRIES, LLM_MODE, exc, delay)
                await asyncio.sleep(delay)
    logger.error("All retries exhausted (mode=%s): %s", LLM_MODE, last_exc)
    return ""


async def llm_stream(
    prompt: str,
    system: str = "You are a helpful assistant.",
    agent: str = "",
) -> AsyncGenerator[str, None]:
    """Streaming call. Yields tokens as they arrive from the API.

    Args:
        prompt: The user message.
        system: System prompt.
        agent:  Optional agent name for per-agent model routing.
    """
    model = _resolve_model(agent)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]

    with _usage_lock:
        _usage_stats["total_calls"] += 1

    # ── LiteLLM mode (default) ──
    if LLM_MODE == "litellm":
        try:
            await _rate_limit_acquire()
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": _TEMPERATURE,
                "stream": True,
            }
            if _SEED is not None:
                kwargs["seed"] = _SEED
            stream = await litellm.acompletion(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.error("LLM streaming failed (model=%s): %s", model, exc)
            yield ""
        return

    # ── OpenAI-compatible client ──
    client = _get_client()
    if client:
        try:
            await _rate_limit_acquire()
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": _TEMPERATURE,
                "stream": True,
            }
            if _SEED is not None:
                kwargs["seed"] = _SEED
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.error("LLM streaming failed (mode=%s): %s", LLM_MODE, exc)
            yield ""
        return

    logger.error("Invalid LLM_MODE: %s", LLM_MODE)
    yield ""


# ── Preflight ──────────────────────────────────────────────────────────────

async def preflight() -> tuple[bool, str]:
    """Quick connectivity test. Returns (ok, message).

    Call this before starting a long pipeline run to fail fast
    if the API key or model is misconfigured.
    """
    model = _resolve_model("")
    try:
        result = await llm(
            "Reply with exactly: OK",
            system="You are a test assistant. Reply with exactly one word.",
            agent="",
        )
        if result.strip():
            return True, f"OK — model={model}, mode={LLM_MODE}"
        return False, f"Empty response from model={model}"
    except Exception as exc:
        return False, f"Preflight failed (model={model}): {exc}"
