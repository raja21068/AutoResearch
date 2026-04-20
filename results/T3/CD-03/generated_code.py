"""Generated code for RL CartPole with Ablation"""

# Task: CD-03 — RL CartPole with Ablation
# Domain: rl

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for RL CartPole with Ablation."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
