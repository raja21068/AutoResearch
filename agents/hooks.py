"""
agents/hooks.py — Session lifecycle hooks.

Events: session_start, session_end, before_execution, after_execution,
        before_commit, after_test, on_error
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    BEFORE_EXECUTION = "before_execution"
    AFTER_EXECUTION = "after_execution"
    BEFORE_COMMIT = "before_commit"
    AFTER_TEST = "after_test"
    ON_ERROR = "on_error"
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"


HookFn = Callable[[dict], Awaitable[None]]


@dataclass
class HookRecord:
    event: HookEvent
    timestamp: float
    data: dict


class HookManager:
    """Manages lifecycle hooks for the agent pipeline."""

    def __init__(self):
        self._hooks: dict[HookEvent, list[HookFn]] = {e: [] for e in HookEvent}
        self._history: list[HookRecord] = []

    def register(self, event: HookEvent, fn: HookFn) -> None:
        self._hooks[event].append(fn)

    async def fire(self, event: HookEvent, data: dict = None) -> None:
        data = data or {}
        data["_event"] = event.value
        data["_timestamp"] = time.time()
        self._history.append(HookRecord(event=event, timestamp=time.time(), data=data))
        for fn in self._hooks[event]:
            try:
                await fn(data)
            except Exception as e:
                logger.warning("Hook %s failed: %s", event.value, e)

    def get_history(self, event: HookEvent = None) -> list[dict]:
        records = self._history
        if event:
            records = [r for r in records if r.event == event]
        return [{"event": r.event.value, "ts": r.timestamp, **r.data} for r in records[-50:]]

    @property
    def registered_count(self) -> int:
        return sum(len(v) for v in self._hooks.values())


# Default hooks
async def _log_hook(data: dict):
    logger.info("Hook [%s]: %s", data.get("_event", "?"), {k: v for k, v in data.items() if not k.startswith("_")})


_manager: HookManager | None = None

def get_hook_manager() -> HookManager:
    global _manager
    if _manager is None:
        _manager = HookManager()
        # Register default logging hook for key events
        _manager.register(HookEvent.SESSION_START, _log_hook)
        _manager.register(HookEvent.ON_ERROR, _log_hook)
    return _manager
