"""Generated code for RAG System with Evaluation"""

# Task: CD-02 — RAG System with Evaluation
# Domain: nlp

import torch
import torch.nn as nn

class Model(nn.Module):
    """Auto-generated model for RAG System with Evaluation."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
