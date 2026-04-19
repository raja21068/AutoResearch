"""Unified registry: find any agent by name."""

from agents.engineering import PlannerAgent, CoderAgent, TesterAgent, DebuggerAgent, CriticAgent
from agents.research import ResearcherAgent, ExperimentAgent, PaperWriterAgent
from agents.memory import MemoryAgent
from agents.gan import GANHarness


class Registry:
    """Single lookup for all agents."""
    def __init__(self):
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.tester = TesterAgent()
        self.debugger = DebuggerAgent()
        self.critic = CriticAgent()
        self.memory = MemoryAgent()
        self.researcher = ResearcherAgent()
        self.experiment = ExperimentAgent()
        self.paper_writer = PaperWriterAgent()
        self.gan = GANHarness()

        self._map = {
            "planner": self.planner, "coder": self.coder, "tester": self.tester,
            "debugger": self.debugger, "critic": self.critic, "memory": self.memory,
            "researcher": self.researcher, "experiment": self.experiment,
            "paper_writer": self.paper_writer, "gan": self.gan,
        }

    def get(self, name):
        if name in self._map:
            return self._map[name]
        try:
            from skills.loader import get_agent_registry
            return get_agent_registry().get(name)
        except Exception:
            return None

    def list_all(self):
        skill_names = []
        try:
            from skills.loader import get_agent_registry
            skill_names = get_agent_registry().list_all()
        except Exception:
            pass
        return {"core": list(self._map.keys()), "skills": skill_names}

    @property
    def total(self):
        try:
            from skills.loader import get_agent_registry
            return len(self._map) + len(get_agent_registry())
        except Exception:
            return len(self._map)


_registry = None
def get_registry():
    global _registry
    if _registry is None: _registry = Registry()
    return _registry
