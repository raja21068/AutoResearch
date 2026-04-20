"""Generated code for Multi-Agent Debate System"""

# Task: CD-10 — Multi-Agent Debate System
# Domain: agents

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for Multi-Agent Debate System."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
