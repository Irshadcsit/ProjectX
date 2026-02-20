"""
models/resnet.py
----------------
ResNet-18 fine-tuned for binary pneumonia classification.

Architecture justification:
  - Residual connections address vanishing gradients in deeper networks
  - ResNet-18 is the lightest ResNet variant — fast training
  - Well-studied on medical imaging tasks → strong reference point
  
"""

import torch
import torch.nn as nn
from torchvision import models, transforms


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 adapted for binary chest X-ray classification.
    """
    MODEL_NAME = "ResNet-18"

    def __init__(self, pretrained: bool = True, dropout: float = 0.4,
                 freeze_blocks: int = 2):
        super().__init__()
        weights      = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone     = models.resnet18(weights=weights)
        self.upsample = transforms.Resize((224, 224), antialias=True)

        # Feature extractor: everything except the final FC
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)

        in_feat       = backbone.fc.in_features   # 512
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feat, 1),
        )
        self._freeze_layers(freeze_blocks)

    def _freeze_layers(self, n_layers: int):
        """Freeze the first n_layers children of ResNet."""
        children = list(self.features.children())
        for child in children[:n_layers]:
            for p in child.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.upsample(x)          # (B,1,28,28) → (B,1,224,224)
        x = x.repeat(1, 3, 1, 1)     # grayscale → pseudo-RGB
        x = self.features(x)          # (B, 512, 1, 1)
        x = torch.flatten(x, 1)       # (B, 512)
        return self.classifier(x)    # raw logit (B,1)
