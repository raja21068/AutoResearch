"""Generated code for ViT Fine-tuning Pipeline"""

# Task: CD-01 — ViT Fine-tuning Pipeline
# Domain: vision

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for ViT Fine-tuning Pipeline."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
