"""
Train SensorLM encoder from scratch on MESA PSG for 5-class sleep staging.

Same training setup as BIOT (lr=1e-4, AdamW, early stopping patience=20,
latest.pth + best.pth checkpointing with resume logic).

Usage:
    python scripts/finetune_sensorlm.py --modality EEG_ONLY
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

sys.path.insert(0, os.path.join(REPO_ROOT, "sleepfm"))
from experiment_paths import new_experiment, experiment_from_checkpoint_dir, link_checkpoint_and_results

from sensorlm_dataset import MODALITY_CHANNELS, SensorLMSleepDataset, HDF5_DIR
from sensorlm_model import SensorLMEncoder

SPLIT_PATH = os.path.join(
    REPO_ROOT, "sleepfm/configs/dataset_split_fromscratch_staging.json"
)


def compute_class_weights(dataset):
    labels = np.array([dataset.index[i][2] for i in range(len(dataset))])
    counts = np.bincount(labels, minlength=5).astype(float)
    weights = 1.0 / np.where(counts > 0, counts, 1.0)
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for x, y, _ in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return f1_score(all_targets, all_preds, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True,
                        choices=list(MODALITY_CHANNELS.keys()))
    parser.add_argument("--fold_key",    default="fold_0")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--hdf5_dir", type=str, default=HDF5_DIR,
                         help="Override the MESA HDF5 directory (default: "
                              "sensorlm_dataset.py's built-in 350-subject "
                              "Puhti-era path). Pass the full-cohort path "
                              "for full-cohort fine-tuning.")
    parser.add_argument("--split_id", type=str, default="fold10_v1",
                         help="Selects which split file to use -- resolves to "
                              "sleepfm/configs/dataset_split_{split_id}.json "
                              "(e.g. 'fold5_v1' -> dataset_split_fold5_v1.json). "
                              "Also used for the full_cohort experiment naming scheme.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                         help="Resume an existing full_cohort experiment "
                              "folder (e.g. after a SLURM timeout) instead of "
                              "starting a new timestamped run.")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_num = int(args.fold_key.replace("fold_", ""))
    if args.checkpoint_dir:
        exp = experiment_from_checkpoint_dir(args.checkpoint_dir)
    else:
        exp = new_experiment(model="sensorlm", modality=args.modality,
                              pretrain_method="finetuned", split_id=args.split_id)
    out_dir = str(exp.fold_dir(fold_num))
    os.makedirs(out_dir, exist_ok=True)
    print(f">>> OUTPUT PATH: {out_dir}", flush=True)
    print(f">>> RESULTS DIR: {exp.results_dir}", flush=True)

    # Derived from exp.split_id (not args.split_id directly) so a resumed
    # run (--checkpoint_dir) keeps using the split file it actually started
    # with, even if --split_id isn't re-passed on resume.
    split_path = os.path.join(REPO_ROOT, f"sleepfm/configs/dataset_split_{exp.split_id}.json")
    train_ds = SensorLMSleepDataset(split_path, "train", args.modality, args.fold_key,
                                     hdf5_dir=args.hdf5_dir)
    val_ds   = SensorLMSleepDataset(split_path, "val",   args.modality, args.fold_key,
                                     hdf5_dir=args.hdf5_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"[{args.modality}] train={len(train_ds)}  val={len(val_ds)}", flush=True)

    n_channels = len(MODALITY_CHANNELS[args.modality])
    model      = SensorLMEncoder(n_channels=n_channels).to(device)
    n_params   = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Resume logic
    start_epoch     = 0
    best_val_f1     = -1.0
    best_epoch      = 0
    patience_counter = 0

    latest_path = os.path.join(out_dir, "latest.pth")
    if os.path.exists(latest_path):
        # weights_only=False: best_val_f1 (sklearn f1_score, a numpy.float64)
        # is embedded in this dict, which PyTorch 2.6+'s weights_only=True
        # default rejects. Safe here -- our own checkpoint, not external data.
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_f1      = ckpt["best_val_f1"]
        best_epoch       = ckpt["best_epoch"]
        patience_counter = ckpt["patience_counter"]
        print(f"Resumed from epoch {start_epoch} (best val F1={best_val_f1:.4f})", flush=True)
    else:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_f1     = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_f1={val_f1:.4f}  best={best_val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
            link_checkpoint_and_results(exp, fold_num)
        else:
            patience_counter += 1

        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1":          best_val_f1,
            "best_epoch":           best_epoch,
            "patience_counter":     patience_counter,
        }, latest_path)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch+1})", flush=True)
            break

    print(f"Training done. Best val F1={best_val_f1:.4f} at epoch {best_epoch+1}", flush=True)


if __name__ == "__main__":
    main()
