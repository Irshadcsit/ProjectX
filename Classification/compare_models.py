"""
task1_classification/compare_models.py
----------------------------------------
Loads test_metrics.json from each model's output folder and generates
a side-by-side comparison table + bar chart.

Usage:
    python compare_models.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODELS      = ["efficientnet", "resnet", "custom_cnn"]
METRICS     = ["accuracy", "precision", "recall", "f1", "auc"]
COLORS      = ["#2196F3", "#4CAF50", "#FF9800"]
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_metrics(model_key: str) -> dict | None:
    path = os.path.join(OUTPUT_DIR, model_key, "test_metrics.json")
    if not os.path.exists(path):
        print(f"  ⚠ No results for '{model_key}' — run evaluate.py --model {model_key} first")
        return None
    with open(path) as f:
        return json.load(f)


def print_comparison_table(all_metrics: dict):
    models  = list(all_metrics.keys())
    header  = f"{'Metric':<14}" + "".join(f"{m:>20}" for m in models)
    divider = "-" * (14 + 20 * len(models))
    print(f"\n{'='*len(header)}")
    print("  MODEL COMPARISON — TEST SET")
    print(f"{'='*len(header)}")
    print(header)
    print(divider)
    for metric in METRICS:
        row = f"{metric.capitalize():<14}"
        best_val = max(all_metrics[m].get(metric, 0) for m in models)
        for m in models:
            val  = all_metrics[m].get(metric, float("nan"))
            star = " *" if abs(val - best_val) < 1e-6 else "  "
            row += f"{val:>18.4f}{star}"
        print(row)
    print(divider)
    print("* = best value for this metric\n")


def plot_comparison(all_metrics: dict, out_path: str):
    models  = list(all_metrics.keys())
    n_m     = len(METRICS)
    n_mod   = len(models)
    x       = np.arange(n_m)
    width   = 0.8 / n_mod

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (model, color) in enumerate(zip(models, COLORS[:n_mod])):
        vals  = [all_metrics[model].get(m, 0) for m in METRICS]
        bars  = ax.bar(x + i * width - (n_mod - 1) * width / 2,
                       vals, width * 0.9, label=all_metrics[model].get("model_name", model),
                       color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in METRICS], fontsize=12)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — PneumoniaMNIST Test Set",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", lw=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Comparison chart saved: {out_path}")


def plot_roc_comparison(all_metrics: dict, out_path: str):
    """Overlay ROC curves from each model's saved roc data (if available)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")

    for model, color in zip(all_metrics.keys(), COLORS):
        m    = all_metrics[model]
        auc  = m.get("auc", None)
        name = m.get("model_name", model)
        if auc:
            # We only have the AUC scalar here; draw reference label
            ax.plot([], [], color=color, lw=2, label=f"{name}  AUC={auc:.4f}")

    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("AUC Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ AUC comparison saved: {out_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_metrics = {}
    for m in MODELS:
        data = load_metrics(m)
        if data:
            all_metrics[m] = data

    if not all_metrics:
        print("No trained models found. Train at least one model first.")
        sys.exit(1)

    print_comparison_table(all_metrics)
    plot_comparison(all_metrics,
        os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plot_roc_comparison(all_metrics,
        os.path.join(OUTPUT_DIR, "auc_comparison.png"))

    # Save combined JSON
    with open(os.path.join(OUTPUT_DIR, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ✓ Combined metrics: {os.path.join(OUTPUT_DIR, 'all_metrics.json')}")
