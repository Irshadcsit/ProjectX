"""
task1_classification/train.py
------------------------------
Training script. Swap models by changing --model argument.
Results saved to outputs/<model_name>/ for easy comparison.

Usage:
    python train.py --model efficientnet --epochs 20
    python train.py --model resnet       --epochs 20
    python train.py --model custom_cnn  --epochs 30 --lr 3e-4

Two-phase training strategy:
  Phase 1 (warm-up): Train only unfrozen layers with high LR
  Phase 2 (fine-tune): Unfreeze all layers, lower LR + cosine annealing
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, time, json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import get_dataloaders, get_class_weights
from models.registry import get_model, get_model_name, count_parameters


def get_output_dir(model_name: str, base: str = "outputs") -> str:
    """Each model gets its own output folder for easy comparison."""
    out = os.path.join(os.path.dirname(__file__), "..", base, model_name)
    os.makedirs(out, exist_ok=True)
    return out


def set_seed(seed: int = 42):
    import numpy as np, random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)
            logits = model(images)
            loss   = criterion(logits.squeeze(1), labels.squeeze(1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels.squeeze(1).long()).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers)
    pos_weight, n_neg, n_pos = get_class_weights()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(args.model, pretrained=not args.no_pretrain,
                      dropout=args.dropout).to(device)
    model_name = get_model_name(model)
    params     = count_parameters(model)
    output_dir = get_output_dir(args.model)
    print(f"[Model] {model_name}  →  total={params['total']:,} | "
          f"trainable={params['trainable']:,} | frozen={params['frozen']:,}")
    print(f"[Output] Saving to: {output_dir}")

    # ── Loss & Optimiser ──────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"model": args.model, "model_name": model_name,
               "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc    = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        # Phase 2 — unfreeze all layers
        if epoch == args.unfreeze_epoch:
            model.unfreeze_all()
            optimizer = AdamW(model.parameters(),
                              lr=args.lr * 0.1, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=args.epochs - epoch, eta_min=1e-7)
            print(f"\n[Epoch {epoch}] Unfreezing all layers (LR={args.lr*0.1:.2e})")

        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, False)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr)

        flag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"epoch": epoch, "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": va_acc, "args": vars(args)}, best_model_path)
            flag = " ← best"

        print(f"Epoch [{epoch:02d}/{args.epochs}]  "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
              f"Val loss={va_loss:.4f} acc={va_acc:.4f}  |  "
              f"LR={lr:.2e}  |  {time.time()-t0:.1f}s{flag}")

    # ── Save history ──────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Best val_acc={best_val_acc:.4f} | Model: {best_model_path}")
    return history, best_model_path


def parse_args():
    p = argparse.ArgumentParser(description="Train pneumonia classifier")
    p.add_argument("--model",          type=str,   default="efficientnet",
                   choices=["efficientnet", "resnet", "custom_cnn"],
                   help="Which model to train")
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--dropout",        type=float, default=0.4)
    p.add_argument("--unfreeze_epoch", type=int,   default=6)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no_pretrain",    action="store_true",
                   help="Disable pretrained weights (always off for custom_cnn)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
