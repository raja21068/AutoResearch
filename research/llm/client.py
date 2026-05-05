"""LLMClient — provider-agnostic chat interface for the research pipeline.

Wraps LiteLLM so that *any* provider (OpenAI, Gemini, Anthropic, DeepSeek,
OpenRouter, Ollama, Azure, etc.) works without code changes.  Only
``config.arc.yaml`` or ``.env`` values need to change.

Features
--------
* Automatic retry with exponential back-off
* In-memory response cache (dedup identical prompts within a run)
* Token-bucket rate limiter (configurable RPM)
* Cost / token tracking
* ``preflight()`` for quick connectivity check before long runs
* ``<think>`` tag stripping for reasoning-model compatibility
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import litellm

# Silence LiteLLM's chatty success logs
litellm.set_verbose = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "minimax": "https://api.minimaxi.com/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
}


def _think_strip(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Simple token-bucket rate limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Thread-safe token-bucket that enforces *rpm* requests per minute."""

    def __init__(self, rpm: int = 0):
        self._rpm = rpm
        self._tokens = float(rpm)
        self._last = time.monotonic()
        self._lock = Lock()

    def acquire(self) -> None:
        if self._rpm <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._rpm, self._tokens + elapsed * (self._rpm / 60.0))
            if self._tokens < 1:
                wait = (1 - self._tokens) / (self._rpm / 60.0)
                time.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


# ---------------------------------------------------------------------------
# Usage tracker
# ---------------------------------------------------------------------------


@dataclass
class UsageStats:
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    cache_hits: int = 0
    retries: int = 0


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Provider-agnostic synchronous chat client used by the research pipeline.

    All research pipeline code calls ``client.chat(messages, system=…)``.
    Internally this delegates to LiteLLM which routes to the configured
    provider transparently.
    """

    def __init__(
        self,
        *,
        provider: str = "openai",
        base_url: str = "",
        api_key: str = "",
        primary_model: str = "",
        fallback_models: list[str] | None = None,
        wire_api: str = "chat_completions",
        temperature: float = 0.0,
        max_retries: int = 3,
        rpm_limit: int = 0,
        enable_cache: bool = True,
    ) -> None:
        self.provider = provider.lower()
        self.base_url = base_url or _PROVIDER_BASE_URLS.get(self.provider, "")
        self.api_key = api_key
        self.primary_model = primary_model or self._default_model()
        self.fallback_models = fallback_models or []
        self.wire_api = wire_api
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_cache = enable_cache

        self._rate_limiter = _RateLimiter(rpm_limit)
        self._cache: dict[str, str] = {}
        self._cache_lock = Lock()
        self.stats = UsageStats()

        # Push API key into the env var that LiteLLM expects.
        self._setup_env()

    # -- env setup ---------------------------------------------------------

    def _setup_env(self) -> None:
        """Set provider-specific env vars so LiteLLM can authenticate."""
        if not self.api_key:
            return
        mapping: dict[str, list[str]] = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "together": ["TOGETHER_API_KEY", "TOGETHERAI_API_KEY"],
            "fireworks": ["FIREWORKS_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
        }
        for env_var in mapping.get(self.provider, []):
            os.environ.setdefault(env_var, self.api_key)

    # -- default model per provider ----------------------------------------

    def _default_model(self) -> str:
        defaults: dict[str, str] = {
            "openai": "gpt-4o",
            "anthropic": "anthropic/claude-sonnet-4-5",
            "deepseek": "deepseek/deepseek-chat",
            "gemini": "gemini/gemini-2.5-flash",
            "google": "gemini/gemini-2.5-flash",
            "openrouter": "openrouter/auto",
            "groq": "groq/llama-3.3-70b-versatile",
            "together": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "ollama": "ollama/llama3.1",
            "mistral": "mistral/mistral-large-latest",
            "local": "openai/local-model",
        }
        return defaults.get(self.provider, "gpt-4o")

    # -- model name normalisation ------------------------------------------

    def _resolve_model(self, model: str | None = None) -> str:
        """Return a LiteLLM-compatible model string."""
        m = model or self.primary_model
        # If the model already has a provider prefix, use as-is
        if "/" in m:
            return m
        # Auto-prefix based on provider so LiteLLM routes correctly
        prefix_map: dict[str, str] = {
            "deepseek": "deepseek",
            "gemini": "gemini",
            "google": "gemini",
            "anthropic": "anthropic",
            "groq": "groq",
            "together": "together_ai",
            "ollama": "ollama",
            "mistral": "mistral",
            "cohere": "cohere",
            "azure": "azure",
            "bedrock": "bedrock",
            "vertex_ai": "vertex_ai",
            "huggingface": "huggingface",
            "perplexity": "perplexity",
            "fireworks": "fireworks_ai",
            "xai": "xai",
        }
        prefix = prefix_map.get(self.provider)
        if prefix:
            return f"{prefix}/{m}"
        return m

    # -- cache key ---------------------------------------------------------

    @staticmethod
    def _cache_key(model: str, messages: list[dict], system: str, json_mode: bool) -> str:
        blob = json.dumps({"m": model, "msgs": messages, "sys": system, "jm": json_mode},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- core chat ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = "",
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        strip_thinking: bool = True,
    ) -> str:
        """Send a chat completion request. Returns the assistant's text."""
        resolved = self._resolve_model(model)
        temp = temperature if temperature is not None else self.temperature

        # -- cache check ---------------------------------------------------
        ck = self._cache_key(resolved, messages, system, json_mode)
        if self.enable_cache:
            with self._cache_lock:
                if ck in self._cache:
                    self.stats.cache_hits += 1
                    return self._cache[ck]

        # -- build messages with system prompt -----------------------------
        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # -- build kwargs for litellm.completion ---------------------------
        kwargs: dict[str, Any] = {
            "model": resolved,
            "messages": full_messages,
            "temperature": temp,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.base_url and self.provider not in ("openai",):
            kwargs["api_base"] = self.base_url

        # -- call with retry + fallback ------------------------------------
        models_to_try = [resolved] + [self._resolve_model(f) for f in self.fallback_models]
        last_exc: Exception | None = None

        for model_name in models_to_try:
            kwargs["model"] = model_name
            for attempt in range(1 + self.max_retries):
                try:
                    self._rate_limiter.acquire()
                    response = litellm.completion(**kwargs)
                    text = response.choices[0].message.content or ""

                    # -- track usage -------------------------------------------
                    self.stats.total_calls += 1
                    usage = getattr(response, "usage", None)
                    if usage:
                        self.stats.total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                        self.stats.total_completion_tokens += getattr(usage, "completion_tokens", 0)
                    try:
                        cost = litellm.completion_cost(completion_response=response)
                        self.stats.total_cost_usd += cost
                    except Exception:
                        pass

                    # -- strip thinking tags -----------------------------------
                    if strip_thinking:
                        text = _think_strip(text)

                    # -- cache -------------------------------------------------
                    if self.enable_cache:
                        with self._cache_lock:
                            self._cache[ck] = text

                    return text

                except Exception as exc:
                    last_exc = exc
                    self.stats.retries += 1
                    err_str = str(exc)

                    # Disable json_mode on 400 — provider may not support it
                    if json_mode and "400" in err_str:
                        logger.warning("HTTP 400 with json_mode — disabling for retry.")
                        kwargs.pop("response_format", None)

                    if attempt < self.max_retries:
                        delay = min(2 ** (attempt + 1), 30)
                        logger.warning(
                            "LLM call failed (%s, attempt %d/%d): %s — retrying in %ds",
                            model_name, attempt + 1, 1 + self.max_retries, exc, delay,
                        )
                        time.sleep(delay)

            logger.warning("All retries exhausted for model %s, trying fallback…", model_name)

        raise RuntimeError(f"All models failed. Last error: {last_exc}") from last_exc

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> tuple[bool, str]:
        """Quick connectivity check. Returns (ok, message)."""
        try:
            resp = self.chat(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                system="You are a test assistant. Reply with exactly one word.",
                max_tokens=10,
            )
            if resp.strip():
                return True, f"OK — model={self.primary_model}, provider={self.provider}"
            return False, "Empty response from model"
        except Exception as exc:
            return False, f"Preflight failed: {exc}"

    # -- convenience -------------------------------------------------------

    def usage_summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.stats.total_calls,
            "prompt_tokens": self.stats.total_prompt_tokens,
            "completion_tokens": self.stats.total_completion_tokens,
            "estimated_cost_usd": round(self.stats.total_cost_usd, 4),
            "cache_hits": self.stats.cache_hits,
            "retries": self.stats.retries,
        }
