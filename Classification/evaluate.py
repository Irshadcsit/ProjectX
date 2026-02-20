"""
task1_classification/evaluate.py
---------------------------------
Evaluation script. Loads a trained model, runs inference on the test set,
and generates all required metrics + visualizations.
Results saved to outputs/<model_name>/ for side-by-side comparison.

Usage:
    python evaluate.py --model efficientnet
    python evaluate.py --model resnet
    python evaluate.py --model custom_cnn
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report
)

from data.dataset import get_dataloaders, CLASS_NAMES
from models.registry import get_model, get_model_name


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_output_dir(model_key: str) -> str:
    out = os.path.join(os.path.dirname(__file__), "..", "outputs", model_key)
    os.makedirs(out, exist_ok=True)
    return out


def denorm(t, mean=0.5, std=0.5):
    return (t * std + mean).clamp(0, 1)


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    probs_l, preds_l, labels_l, images_l = [], [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        probs_l.extend(probs.tolist())
        preds_l.extend(preds.tolist())
        labels_l.extend(labels.squeeze(1).numpy().astype(int).tolist())
        images_l.extend(images.cpu().unbind(0))
    return (np.array(probs_l), np.array(preds_l),
            np.array(labels_l), images_l)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(probs, preds, labels):
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc":       float(roc_auc_score(labels, probs)),
    }


def print_metrics(metrics: dict, model_name: str):
    print(f"\n{'='*46}")
    print(f"  TEST RESULTS — {model_name}")
    print(f"{'='*46}")
    for k, v in metrics.items():
        print(f"  {k.capitalize():12s}: {v:.4f}")
    print(f"{'='*46}\n")


# ── Plot functions ────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, model_name, out_dir):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j+0.5, i+0.72, f"({cm_norm[i,j]*100:.1f}%)",
                    ha="center", fontsize=9, color="gray")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


def plot_roc_curve(labels, probs, auc, model_name, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


def plot_training_curves(out_dir, model_name):
    hist_path = os.path.join(out_dir, "training_history.json")
    if not os.path.exists(hist_path):
        print(f"  ⚠ No training_history.json found in {out_dir}")
        return
    with open(hist_path) as f:
        h = json.load(f)
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, h["train_loss"], label="Train", color="#E91E63", lw=2)
    axes[0].plot(epochs, h["val_loss"],   label="Val",   color="#2196F3", lw=2)
    axes[0].set_title("BCE Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, h["train_acc"], label="Train", color="#E91E63", lw=2)
    axes[1].plot(epochs, h["val_acc"],   label="Val",   color="#2196F3", lw=2)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Acc")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


def plot_failure_cases(images, labels, preds, probs, model_name, out_dir, n_each=6):
    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    fn_idx = np.where((preds == 0) & (labels == 1))[0]
    print(f"\n  Failure Analysis:")
    print(f"    False Positives (Normal→Pneumonia): {len(fp_idx)}")
    print(f"    False Negatives (Pneumonia→Normal): {len(fn_idx)}  ← clinically dangerous")

    n_fp  = min(len(fp_idx), n_each)
    n_fn  = min(len(fn_idx), n_each)
    ncols = max(n_fp, n_fn, 1)

    fig, axes = plt.subplots(2, ncols, figsize=(ncols * 2, 5))
    if axes.ndim == 1: axes = axes[np.newaxis, :]

    for col in range(n_fp):
        idx = fp_idx[col]
        axes[0, col].imshow(denorm(images[idx]).squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"Conf: {probs[idx]:.2f}", fontsize=8, color="#F44336")
        axes[0, col].axis("off")
    if n_fp > 0:
        axes[0, 0].set_ylabel(f"FP (Normal→Pneumonia)\nn={len(fp_idx)}",
                              fontsize=9, color="#F44336", fontweight="bold")

    for col in range(n_fn):
        idx = fn_idx[col]
        axes[1, col].imshow(denorm(images[idx]).squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"Conf: {probs[idx]:.2f}", fontsize=8, color="#FF9800")
        axes[1, col].axis("off")
    if n_fn > 0:
        axes[1, 0].set_ylabel(f"FN (Pneumonia→Normal)\nn={len(fn_idx)}",
                              fontsize=9, color="#FF9800", fontweight="bold")

    for r in range(2):
        for c in range(max(n_fp, n_fn), ncols):
            axes[r, c].axis("off")

    plt.suptitle(f"Failure Cases — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "failure_cases.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


def plot_confidence_distribution(probs, labels, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 30)
    ax.hist(probs[labels==0], bins=bins, alpha=0.6, color="#4CAF50",
            label="True Normal", density=True)
    ax.hist(probs[labels==1], bins=bins, alpha=0.6, color="#F44336",
            label="True Pneumonia", density=True)
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold (0.5)")
    ax.set_xlabel("Predicted Probability (Pneumonia)")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence Distribution — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "confidence_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


def plot_sample_predictions(images, labels, preds, probs, model_name, out_dir, n=12):
    indices = np.random.choice(len(labels), size=min(n, len(labels)), replace=False)
    ncols   = 6
    nrows   = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2.5))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        correct = preds[idx] == labels[idx]
        color   = "#2196F3" if correct else "#F44336"
        axes[i].imshow(denorm(images[idx]).squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(
            f"GT:{CLASS_NAMES[labels[idx]]}\nP:{CLASS_NAMES[preds[idx]]}({probs[idx]:.2f})",
            fontsize=7, color=color)
        axes[i].axis("off")
    for j in range(len(indices), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Sample Predictions — {model_name} (Blue=Correct, Red=Wrong)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "sample_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(args):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir    = get_output_dir(args.model)
    model_path = os.path.join(out_dir, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run train.py first.")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(model_path, map_location=device)
    model = get_model(args.model, pretrained=False, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model_name = get_model_name(model)
    print(f"[Eval] Loaded {model_name} from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    # ── Data ──────────────────────────────────────────────────────────────────
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2)

    # ── Inference ─────────────────────────────────────────────────────────────
    probs, preds, labels, images = run_inference(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(probs, preds, labels)
    metrics["model"]      = args.model
    metrics["model_name"] = model_name
    print_metrics(metrics, model_name)
    print("[sklearn report]\n", classification_report(labels, preds,
          target_names=CLASS_NAMES, digits=4))

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Plots] Generating visualizations...")
    plot_training_curves(out_dir, model_name)
    plot_confusion_matrix(labels, preds, model_name, out_dir)
    plot_roc_curve(labels, probs, metrics["auc"], model_name, out_dir)
    plot_failure_cases(images, labels, preds, probs, model_name, out_dir)
    plot_confidence_distribution(probs, labels, model_name, out_dir)
    plot_sample_predictions(images, labels, preds, probs, model_name, out_dir)

    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained pneumonia classifier")
    p.add_argument("--model",      type=str, default="efficientnet",
                   choices=["efficientnet", "resnet", "custom_cnn"])
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
