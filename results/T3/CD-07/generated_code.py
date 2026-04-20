"""Generated code for AutoML Hyperparameter Search"""

# Task: CD-07 — AutoML Hyperparameter Search
# Domain: ml

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for AutoML Hyperparameter Search."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
