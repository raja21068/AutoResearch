"""Domain-specific prompt adapters.

Each adapter customizes prompt blocks for a specific research domain
while the ML adapter preserves existing behavior unchanged.
"""

from research.domains.adapters.ml import MLPromptAdapter
from research.domains.adapters.generic import GenericPromptAdapter
from research.domains.adapters.physics import PhysicsPromptAdapter
from research.domains.adapters.economics import EconomicsPromptAdapter
from research.domains.adapters.biology import BiologyPromptAdapter
from research.domains.adapters.chemistry import ChemistryPromptAdapter
from research.domains.adapters.neuroscience import NeurosciencePromptAdapter
from research.domains.adapters.robotics import RoboticsPromptAdapter

__all__ = [
    "MLPromptAdapter",
    "GenericPromptAdapter",
    "PhysicsPromptAdapter",
    "EconomicsPromptAdapter",
    "BiologyPromptAdapter",
    "ChemistryPromptAdapter",
    "NeurosciencePromptAdapter",
    "RoboticsPromptAdapter",
]
