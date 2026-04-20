"""Generated code for Time Series Anomaly Detection"""

# Task: CD-09 — Time Series Anomaly Detection
# Domain: ml

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for Time Series Anomaly Detection."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
