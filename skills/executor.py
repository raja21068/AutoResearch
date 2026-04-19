"""
skills/skill_executor.py — Executes loaded skill agents dynamically.

Converts a SkillAgent .md definition into a live LLM call with the
system prompt from the file. Injects language-specific rules when relevant.
"""

import logging
from typing import AsyncGenerator

from llm import get_router
from skills.loader import SkillAgent, get_agent_registry
from skills.engine import get_rule_engine

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "opus": "anthropic/claude-opus-4-5",
    "sonnet": "anthropic/claude-sonnet-4-5",
    "haiku": "anthropic/claude-haiku-4-5",
    "gpt4": "gpt-4o",
    "deepseek": "deepseek/deepseek-chat",
}


class SkillExecutor:
    """Executes any loaded skill agent with context-aware rule injection."""

    def __init__(self):
        self.registry = get_agent_registry()
        self.rules = get_rule_engine()

    async def run(
        self,
        skill_name: str,
        task: str,
        context: str = "",
        language: str = "",
        files: list[str] = None,
    ) -> str:
        """Run a skill agent by name."""
        agent = self.registry.get(skill_name)
        if agent is None:
            available = self.registry.list_all()
            return f"Unknown skill: {skill_name}. Available: {', '.join(available[:20])}"

        # Build system prompt with optional rule injection
        system = agent.system_prompt
        if language:
            rules = self.rules.get_rules(language)
            if rules:
                system += f"\n\n## Language-Specific Rules ({language})\n{rules[:3000]}"
        elif files:
            # Auto-detect language from first file
            for f in files:
                auto_rules = self.rules.get_rules_for_file(f)
                if auto_rules:
                    system += f"\n\n## Detected Rules\n{auto_rules[:3000]}"
                    break

        # Build prompt
        prompt = f"Task: {task}"
        if context:
            prompt += f"\n\nContext:\n{context[:5000]}"
        if files:
            prompt += f"\n\nRelevant files: {', '.join(files[:10])}"

        # Route to appropriate model
        model_name = MODEL_MAP.get(agent.model, agent.model)
        router = get_router()
        return await router.call(prompt, system=system, agent="coder")

    async def stream(
        self,
        skill_name: str,
        task: str,
        context: str = "",
        language: str = "",
    ) -> AsyncGenerator[str, None]:
        """Stream a skill agent's response."""
        agent = self.registry.get(skill_name)
        if agent is None:
            yield f"Unknown skill: {skill_name}"
            return

        system = agent.system_prompt
        if language:
            rules = self.rules.get_rules(language)
            if rules:
                system += f"\n\n## Language Rules\n{rules[:3000]}"

        router = get_router()
        async for token in router.stream(f"Task: {task}\n\n{context[:3000]}", system=system, agent="coder"):
            yield token

    def find_best_agent(self, task: str, language: str = "") -> SkillAgent | None:
        """Auto-select the best skill agent for a task."""
        t = task.lower()
        # Exact matches first
        if "review" in t and language:
            agent = self.registry.get(f"{language}-reviewer")
            if agent:
                return agent
        if "security" in t:
            return self.registry.get("security-reviewer")
        if "build" in t or "compile" in t:
            if language:
                agent = self.registry.get(f"{language}-build-resolver")
                if agent:
                    return agent
            return self.registry.get("build-error-resolver")
        if "architect" in t or "design" in t:
            return self.registry.get("architect")
        if "plan" in t:
            return self.registry.get("planner")
        if "test" in t or "tdd" in t:
            return self.registry.get("tdd-guide")
        if "review" in t:
            return self.registry.get("code-reviewer")
        if "performance" in t or "optimize" in t:
            return self.registry.get("performance-optimizer")
        if "simplif" in t or "refactor" in t:
            return self.registry.get("refactor-cleaner")
        # Fallback
        matches = self.registry.search(task.split()[0] if task else "code")
        return matches[0] if matches else None


_executor: SkillExecutor | None = None

def get_skill_executor() -> SkillExecutor:
    global _executor
    if _executor is None:
        _executor = SkillExecutor()
    return _executor
