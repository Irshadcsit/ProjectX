"""
models/efficientnet.py
----------------------
EfficientNet-B0 fine-tuned for binary pneumonia classification.

Architecture justification:
  - Compound scaling (depth × width × resolution) 
  - ImageNet pretrained: low-level feature detectors transfer well to X-rays
  - Lightweight (~5.3M params) 
  - Two-phase fine-tuning strategy to prevent catastrophic forgetting
"""

import torch
import torch.nn as nn
from torchvision import models, transforms


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 adapted for binary chest X-ray classification.

    Modifications:
      1. Grayscale input (1-ch) replicated to 3-ch to reuse pretrained conv weights
      2. Internal upsample 28×28 → 224×224 to match expected receptive field
      3. Head: 1280-d → Dropout → Linear(1) (raw logit for BCEWithLogitsLoss)
    """
    MODEL_NAME = "EfficientNet-B0"

    def __init__(self, pretrained: bool = True, dropout: float = 0.4,
                 freeze_blocks: int = 4):
        super().__init__()
        weights     = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone    = models.efficientnet_b0(weights=weights)
        self.upsample   = transforms.Resize((224, 224), antialias=True)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        in_feat         = backbone.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, 1),
        )
        self._freeze_blocks(freeze_blocks)

    def _freeze_blocks(self, n: int):
        for i, block in enumerate(self.features):
            if i < n:
                for p in block.parameters():
                    p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.upsample(x)          # (B,1,28,28) → (B,1,224,224)
        x = x.repeat(1, 3, 1, 1)     # grayscale → pseudo-RGB
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)    # raw logit (B,1)
