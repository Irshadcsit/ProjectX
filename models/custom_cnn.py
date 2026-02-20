"""
models/custom_cnn.py
--------------------
Custom lightweight CNN for binary pneumonia classification.

Architecture justification:
  - Baseline model trained from scratch (no pretrained weights)
  - Demonstrates what can be achieved without transfer learning
  - 3 conv blocks → BatchNorm → MaxPool → FC head
"""

import torch
import torch.nn as nn


class CustomCNNClassifier(nn.Module):
    """
    Lightweight 3-block CNN trained from scratch on 28×28 X-ray images.
    No upsampling needed — operates directly at 28×28.
    """
    MODEL_NAME = "Custom-CNN"

    def __init__(self, dropout: float = 0.5, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 28×28 → 14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 14×14
            nn.Dropout2d(0.25),

            # Block 2: 14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 7×7
            nn.Dropout2d(0.25),

            # Block 3: 7×7 → 3×3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),            # 3×3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),          # raw logit
        )

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)      # raw logit (B,1)
