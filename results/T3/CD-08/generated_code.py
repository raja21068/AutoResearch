"""Generated code for Knowledge Graph Construction"""

# Task: CD-08 — Knowledge Graph Construction
# Domain: nlp

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for Knowledge Graph Construction."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
