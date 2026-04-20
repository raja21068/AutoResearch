"""Generated code for Federated Learning Simulation"""

# Task: CD-06 — Federated Learning Simulation
# Domain: distributed

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for Federated Learning Simulation."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
