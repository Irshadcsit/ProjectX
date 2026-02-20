from torchvision import models
import torch.nn as nn

class ViTB16Classifier(nn.Module):
    MODEL_NAME = "ViT-B/16"

    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.vit_b_16(weights=weights)

        in_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

        self.model = backbone

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear")
        x = x.expand(-1, 3, -1, -1)
        return self.model(x)
