"""
data/dataset.py
---------------
Reusable PneumoniaMNIST data pipeline.
Designed to be imported by any notebook or script.

Usage:
    from data.dataset import get_dataloaders, get_datasets, CLASS_NAMES
"""

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST, INFO

# ── Constants (importable by notebooks) ───────────────────────────────────────
DATA_FLAG   = "pneumoniamnist"
INFO_DATA   = INFO[DATA_FLAG]
CLASS_NAMES = list(INFO_DATA["label"].values())   # ["normal", "pneumonia"]
N_CLASSES   = len(CLASS_NAMES)                    # 2
MEAN        = (0.5,)
STD         = (0.5,)


def get_transforms(split: str, image_size: int = 28) -> transforms.Compose:
    """
    Augmentation pipeline per split.

    Medical imaging rationale:
      ✓ Horizontal flip      — lungs are bilaterally symmetric
      ✓ Small rotation ±10°  — patient positioning variance
      ✓ Brightness/contrast  — simulates exposure differences across machines
      ✗ Vertical flip        — anatomically invalid (upside-down chest)
      ✗ Heavy crop           — 28×28 is already tiny; loses diagnostic detail
    """
    if split == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


def get_datasets(download: bool = True, image_size: int = 28):
    """
    Returns (train_ds, val_ds, test_ds) using official MedMNIST splits.
    Fixed splits ensure reproducible benchmarking across experiments.
    """
    train_ds = PneumoniaMNIST(split="train", transform=get_transforms("train", image_size),
                               download=download, size=image_size)
    val_ds   = PneumoniaMNIST(split="val",   transform=get_transforms("val",   image_size),
                               download=download, size=image_size)
    test_ds  = PneumoniaMNIST(split="test",  transform=get_transforms("test",  image_size),
                               download=download, size=image_size)
    return train_ds, val_ds, test_ds


def get_dataloaders(batch_size: int = 64, num_workers: int = 2,
                    download: bool = True, image_size: int = 28):
    """
    Returns (train_loader, val_loader, test_loader).

    Args:
        batch_size:  Mini-batch size
        num_workers: Parallel loading workers (set 0 on Windows)
        download:    Auto-download dataset if missing
        image_size:  Input resolution (28 default; MedMNIST also supports 64, 128, 224)
    """
    train_ds, val_ds, test_ds = get_datasets(download=download, image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[Dataset] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    print(f"[Dataset] classes={CLASS_NAMES} | image_size={image_size}×{image_size}")
    return train_loader, val_loader, test_loader


def get_class_weights(download: bool = True):
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = N_negative / N_positive
    """
    train_ds, _, _ = get_datasets(download=download)
    labels = [int(train_ds[i][1]) for i in range(len(train_ds))]
    n_neg = labels.count(0)
    n_pos = labels.count(1)
    pos_weight = n_neg / n_pos
    print(f"[Dataset] Normal={n_neg} ({n_neg/len(labels)*100:.1f}%) | "
          f"Pneumonia={n_pos} ({n_pos/len(labels)*100:.1f}%) | pos_weight={pos_weight:.3f}")
    return pos_weight, n_neg, n_pos


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch: images={images.shape} labels={labels.shape}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
