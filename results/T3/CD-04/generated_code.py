"""Generated code for GAN Image Generation + FID"""

# Task: CD-04 — GAN Image Generation + FID
# Domain: vision

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for GAN Image Generation + FID."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
