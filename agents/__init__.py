"""
agents/ — ALL agents live here. One import path.

    from agents import PlannerAgent, CoderAgent, GANHarness
    from agents import get_registry, VerificationEngine
"""

from agents.engineering import PlannerAgent, CoderAgent, TesterAgent, DebuggerAgent, CriticAgent
from agents.research import ResearcherAgent, ExperimentAgent, PaperWriterAgent
from agents.memory import MemoryAgent
from agents.gan import GANHarness
from agents.hooks import HookManager, HookEvent, get_hook_manager
from agents.context_modes import ContextMode, get_context_manager
from agents.registry import Registry, get_registry
from agents.conception import ConceptionService
from agents.paper import PaperOrchestrator, PaperAutorater, PlottingAgent
from agents.experiment import ExperimentOrchestrator
from agents.decision import DecisionService
from agents.verification import VerificationEngine

__all__ = [
    "PlannerAgent", "CoderAgent", "TesterAgent", "DebuggerAgent", "CriticAgent",
    "ResearcherAgent", "ExperimentAgent", "PaperWriterAgent", "MemoryAgent",
    "GANHarness", "HookManager", "HookEvent", "get_hook_manager",
    "ContextMode", "get_context_manager", "Registry", "get_registry",
    "ConceptionService", "PaperOrchestrator", "PaperAutorater", "PlottingAgent",
    "ExperimentOrchestrator", "DecisionService", "VerificationEngine",
]
