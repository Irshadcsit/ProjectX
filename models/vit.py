"""
models/vit.py
----------------------
----------------------
Vision Transformer (ViT-B/16) fine-tuned for binary pneumonia classification.

Architecture justification:
  - Self-attention mechanism captures global context in chest X-ray images
  - Patch-based representation (16×16) models long-range spatial dependencies
  - ImageNet pretrained weights improve generalization on limited medical data
  - Dropout and optional freezing help reduce overfitting
"""

import torch
from torchvision import models
import torch.nn as nn

class ViTB16Classifier(nn.Module):
    MODEL_NAME = "ViT-B/16"

    def __init__(self, pretrained=True, dropout=0.4, freeze=True):
        super().__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.vit_b_16(weights=weights)

        in_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

        self.model = backbone

        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear")
        x = x.expand(-1, 3, -1, -1)
        return self.model(x)
