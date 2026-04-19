"""
skills/agent_loader.py — Loads agent definitions from .md files.

Each .md file has YAML frontmatter (name, description, tools, model)
and a markdown body that becomes the agent's system prompt.

Usage:
    registry = AgentRegistry("skills/agents")
    agent = registry.get("code-reviewer")
    prompt = agent.system_prompt
    desc = agent.description
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SkillAgent:
    """A loaded agent definition from a .md skill file."""
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: str = "sonnet"
    color: str = ""
    source_file: str = ""

    @property
    def id(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "tools": self.tools,
            "model": self.model,
        }


def parse_agent_md(filepath: str) -> Optional[SkillAgent]:
    """Parse a .md agent definition file."""
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Cannot read %s: %s", filepath, e)
        return None

    # Split frontmatter from body
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not m:
        return None

    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}

    body = m.group(2).strip()
    name = meta.get("name", Path(filepath).stem)
    desc = meta.get("description", "")
    tools = meta.get("tools", [])
    model = meta.get("model", "sonnet")
    color = meta.get("color", "")

    return SkillAgent(
        name=name,
        description=desc,
        system_prompt=body,
        tools=tools if isinstance(tools, list) else [tools],
        model=model,
        color=color,
        source_file=filepath,
    )


class AgentRegistry:
    """Registry of all loaded skill agents."""

    def __init__(self, agents_dir: str = "skills/agents"):
        self.agents: dict[str, SkillAgent] = {}
        self._dir = agents_dir
        self._load()

    def _load(self):
        d = Path(self._dir)
        if not d.exists():
            logger.warning("Agents dir not found: %s", self._dir)
            return
        for f in sorted(d.glob("*.md")):
            agent = parse_agent_md(str(f))
            if agent:
                self.agents[agent.name] = agent
        logger.info("Loaded %d skill agents from %s", len(self.agents), self._dir)

    def get(self, name: str) -> Optional[SkillAgent]:
        return self.agents.get(name)

    def search(self, query: str) -> list[SkillAgent]:
        """Find agents matching a query (name or description substring)."""
        q = query.lower()
        return [a for a in self.agents.values()
                if q in a.name.lower() or q in a.description.lower()]

    def list_all(self) -> list[str]:
        return sorted(self.agents.keys())

    def by_category(self) -> dict[str, list[SkillAgent]]:
        """Group agents by inferred category."""
        cats: dict[str, list[SkillAgent]] = {
            "reviewers": [], "builders": [], "planners": [],
            "security": [], "ml": [], "other": [],
        }
        for a in self.agents.values():
            n = a.name.lower()
            if "review" in n:
                cats["reviewers"].append(a)
            elif "build" in n or "resolver" in n:
                cats["builders"].append(a)
            elif "plan" in n or "architect" in n:
                cats["planners"].append(a)
            elif "security" in n or "sanitiz" in n:
                cats["security"].append(a)
            elif "gan" in n or "pytorch" in n:
                cats["ml"].append(a)
            else:
                cats["other"].append(a)
        return {k: v for k, v in cats.items() if v}

    def __len__(self):
        return len(self.agents)

    def __contains__(self, name):
        return name in self.agents


# Singleton
_registry: AgentRegistry | None = None

def get_agent_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
