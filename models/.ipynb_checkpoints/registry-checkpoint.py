"""
models/registry.py
------------------
Central model registry. Add new models here to make them available
across all scripts and notebooks without touching other files.

Usage:
    from models.registry import get_model, list_models

    model = get_model("efficientnet")   # swap to "resnet" or "custom_cnn"
    model = get_model("resnet", dropout=0.3)
"""

from .efficientnet import EfficientNetB0Classifier
from .resnet       import ResNet18Classifier
from .custom_cnn   import CustomCNNClassifier
from .vit          import ViTB16Classifier 

# ── Registry: name → class ────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "efficientnet": EfficientNetB0Classifier,
    "resnet":       ResNet18Classifier,
    "custom_cnn":   CustomCNNClassifier,
    "vit":          ViTB16Classifier,
}


def get_model(name: str, **kwargs):
    """
    Instantiate a model by name.

    Args:
        name:    One of the keys in MODEL_REGISTRY
        **kwargs: Passed directly to the model constructor
                  e.g. pretrained=True, dropout=0.4

    Returns:
        Instantiated (untrained) model

    Example:
        model = get_model("efficientnet", pretrained=True, dropout=0.4)
        model = get_model("resnet",       pretrained=True, dropout=0.3)
        model = get_model("custom_cnn",   dropout=0.5)
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """Return list of registered model names."""
    return list(MODEL_REGISTRY.keys())


def get_model_name(model) -> str:
    """Return the display name of a model instance."""
    return getattr(model, "MODEL_NAME", type(model).__name__)


def count_parameters(model) -> dict:
    """Count total, trainable, and frozen parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
