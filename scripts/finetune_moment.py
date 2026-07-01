"""
Linear-probe MOMENT-1-large on MESA PSG for 5-class sleep staging.

Loads AutonLab/MOMENT-1-large with the T5 encoder and patch embedder frozen
(341M params unchanged), trains only the freshly-initialized classification
head (~15K params). This is the standard MOMENT linear-probing protocol.

Usage:
    python scripts/finetune_moment.py --modality EEG_ONLY
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

# Must be set before h5py is imported (via moment_dataset) to take effect.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from moment_dataset import MESADataset, MODALITY_CHANNELS
from momentfm import MOMENTPipeline

SPLIT_PATH = os.path.join(REPO_ROOT, "sleepfm/configs/dataset_split_fromscratch_staging.json")
PRETRAINED_NAME = "AutonLab/MOMENT-1-large"
CKPT_ROOT = "/scratch/project_2019517/moment/checkpoints"

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def build_model(modality, n_classes=5, pretrained_name=PRETRAINED_NAME):
    n_channels = len(MODALITY_CHANNELS[modality])
    model = MOMENTPipeline.from_pretrained(
        pretrained_name,
        model_kwargs={
            "task_name": "classification",
            "n_channels": n_channels,
            "num_class": n_classes,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "reduction": "concat",
        },
    )
    model.init()
    return model


def compute_class_weights(dataset, n_classes=5):
    stages = np.array([s for _, _, s in dataset.index])
    counts = np.bincount(stages, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def forward_logits(model, x):
    mask = torch.ones((x.shape[0], x.shape[-1]), device=x.device)
    out = model(x_enc=x, input_mask=mask)
    return out.logits


def run_epoch(model, loader, device, optimizer=None, class_weights=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = forward_logits(model, x)
            loss = nn.functional.cross_entropy(logits, y, weight=class_weights)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=list(MODALITY_CHANNELS.keys()))
    parser.add_argument("--fold_key", default="fold_0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(CKPT_ROOT, args.modality, args.fold_key)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = MESADataset(SPLIT_PATH, "train", args.modality, fold_key=args.fold_key)
    val_ds = MESADataset(SPLIT_PATH, "validation", args.modality, fold_key=args.fold_key)
    print(f"[{args.modality}] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_model(args.modality).to(device)

    # Freeze everything, then unfreeze only the classification head.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}", flush=True)

    class_weights = compute_class_weights(train_ds).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    start_epoch = 0
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    latest_path = os.path.join(out_dir, "latest.pth")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint["best_val_f1"]
        best_epoch = checkpoint["best_epoch"]
        patience_counter = checkpoint["patience_counter"]
        print(f"Resumed from {latest_path}: starting at epoch {start_epoch + 1}, "
              f"best_val_f1={best_val_f1:.4f} @ epoch {best_epoch}, "
              f"patience_counter={patience_counter}", flush=True)
    else:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_f1 = run_epoch(model, train_loader, device, optimizer, class_weights)
        val_loss, val_f1 = run_epoch(model, val_loader, device, None, class_weights)

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            marker = " *"
        else:
            patience_counter += 1

        print(f"E{epoch + 1:03d} train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}{marker}", flush=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
        }, latest_path)

        if patience_counter >= args.patience:
            print(f"Early stop at epoch {epoch + 1} (best={best_val_f1:.4f} @ epoch {best_epoch})")
            break

    print(f"Done. Best val macro F1 = {best_val_f1:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
