"""
Fine-tune BIOT end-to-end on MESA PSG for 5-class sleep staging.

Loads the BIOT encoder pretrained on EEG-SHHS+PREST (5M MGH resting-EEG
samples + 5M Sleep Heart Health Study samples -- a sleep-PSG cohort
comparable to MESA), attaches a fresh randomly-initialized 5-class head, and
fine-tunes the whole model end-to-end. Saves the checkpoint with the best
validation macro F1 to best.pth.

A second checkpoint, latest.pth, is saved after every epoch (model +
optimizer state + early-stopping counters) so a run killed by SLURM wall
time can be resumed in place: re-running this script with the same
--modality/--fold_key picks up latest.pth automatically and continues from
the next epoch, rather than restarting from scratch.

Usage:
    python scripts/finetune_biot.py --modality EEG_ONLY
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
sys.path.insert(0, "/scratch/project_2019517/BIOT")

# Must be set before h5py is imported (via biot_dataset) to take effect.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from biot_dataset import BIOTSleepDataset, MODALITY_CHANNELS
from model.biot import BIOTClassifier

SPLIT_PATH = os.path.join(REPO_ROOT, "sleepfm/configs/dataset_split_fromscratch_staging.json")
PRETRAINED_PATH = "/scratch/project_2019517/BIOT/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"
# Channel-token table size baked into the pretrained checkpoint. Any
# modality with <=18 channels uses the first len(channels) token slots
# (BIOT's channel tokens are positional IDs, not fixed electrode identities
# -- this is how the BIOT paper itself transfers across differently
# instrumented datasets).
CKPT_BIOT_N_CHANNELS = 18
CKPT_ROOT = "/scratch/project_2019517/biot/checkpoints"

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def build_model(n_classes=5, pretrained_path=PRETRAINED_PATH):
    model = BIOTClassifier(
        emb_size=256, heads=8, depth=4, n_classes=n_classes,
        n_fft=200, hop_length=100, n_channels=CKPT_BIOT_N_CHANNELS,
    )
    state_dict = torch.load(pretrained_path, map_location="cpu")
    model.biot.load_state_dict(state_dict)
    return model


def compute_class_weights(dataset, n_classes=5):
    stages = np.array([s for _, _, s in dataset.index])
    counts = np.bincount(stages, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, device, optimizer=None, class_weights=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pretrained_path", default=PRETRAINED_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(CKPT_ROOT, args.modality, args.fold_key)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = BIOTSleepDataset(SPLIT_PATH, "train", args.modality, fold_key=args.fold_key)
    val_ds = BIOTSleepDataset(SPLIT_PATH, "validation", args.modality, fold_key=args.fold_key)
    print(f"[{args.modality}] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_model(pretrained_path=args.pretrained_path).to(device)
    class_weights = compute_class_weights(train_ds).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
