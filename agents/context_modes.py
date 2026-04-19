"""
agents/context_modes.py — Switchable agent behavior profiles.

Modes: dev (write code), research (explore first), review (critique focus)
Adjusts agent system prompts and tool preferences.
"""

import logging
import os
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ContextMode(str, Enum):
    DEV = "dev"
    RESEARCH = "research"
    REVIEW = "review"
    AUTO = "auto"


_MODE_PROMPTS = {
    ContextMode.DEV: (
        "Mode: Development. Focus on writing, building, implementing. "
        "Produce working code. Bias toward action over analysis."
    ),
    ContextMode.RESEARCH: (
        "Mode: Research. Focus on understanding before acting. "
        "Read widely, explore code/docs, form hypotheses, verify with evidence. "
        "Findings first, recommendations second."
    ),
    ContextMode.REVIEW: (
        "Mode: Review. Focus on critique, quality, correctness. "
        "Find issues, verify logic, check edge cases. "
        "Be thorough but only report issues with >80% confidence."
    ),
}


class ContextModeManager:
    """Manages context modes and loads mode instructions from contexts/ dir."""

    def __init__(self, contexts_dir: str = "skills/contexts"):
        self._dir = Path(contexts_dir)
        self._current = ContextMode.AUTO
        self._custom_modes: dict[str, str] = {}
        self._load_custom()

    def _load_custom(self):
        if not self._dir.exists():
            return
        for f in self._dir.glob("*.md"):
            self._custom_modes[f.stem] = f.read_text(encoding="utf-8", errors="ignore")

    def set_mode(self, mode: str) -> str:
        try:
            self._current = ContextMode(mode.lower())
        except ValueError:
            if mode.lower() in self._custom_modes:
                self._current = ContextMode.DEV  # fallback enum
                return self._custom_modes[mode.lower()]
            return f"Unknown mode: {mode}. Available: {', '.join(self.available())}"
        return f"Mode set to: {self._current.value}"

    def get_mode(self) -> ContextMode:
        return self._current

    def get_prompt_prefix(self) -> str:
        """Get the mode-specific prompt prefix to inject into agents."""
        if self._current in _MODE_PROMPTS:
            return _MODE_PROMPTS[self._current]
        return ""

    def get_full_context(self, mode: str = "") -> str:
        m = mode or self._current.value
        if m in self._custom_modes:
            return self._custom_modes[m]
        return _MODE_PROMPTS.get(ContextMode(m), "")

    def available(self) -> list[str]:
        modes = [m.value for m in ContextMode]
        modes.extend(self._custom_modes.keys())
        return sorted(set(modes))


_ctx_manager: ContextModeManager | None = None

def get_context_manager() -> ContextModeManager:
    global _ctx_manager
    if _ctx_manager is None:
        _ctx_manager = ContextModeManager()
    return _ctx_manager
