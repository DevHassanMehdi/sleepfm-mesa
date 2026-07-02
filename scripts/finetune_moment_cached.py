"""
Train a linear classification head on pre-computed MOMENT embeddings.

Embeddings must be generated first by generate_moment_embeddings.py and live at:
  /scratch/project_2019517/moment/embeddings/{MODALITY}/{fold_key}/{split}/*.npz

Each .npz has keys 'embeddings' (n_epochs, emb_dim) and 'labels' (n_epochs,).
All files for a split are loaded into RAM at dataset init — the largest modality
(BAS_EKG_RESP_EMG, emb_dim=15360) uses ~5 GB for the train set, well within
the 128 GB SLURM allocation.

The head is a single nn.Linear(emb_dim, 5) — identical to MOMENT's own
ClassificationHead linear layer — so results are directly comparable to
finetune_moment.py (linear probing), but epochs run in seconds not hours.

Checkpoints: /scratch/project_2019517/moment/checkpoints/{MODALITY}/{fold_key}/
  latest.pth — saved every epoch; allows resume after wall-time kill
  best.pth   — best validation macro F1; used by evaluate_moment_cached.py

Usage:
    python scripts/finetune_moment_cached.py --modality EEG_ONLY
"""
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from moment_dataset import MODALITY_CHANNELS

SPLIT_PATH = os.path.join(REPO_ROOT, "sleepfm/configs/dataset_split_fromscratch_staging.json")
EMBED_ROOT = "/scratch/project_2019517/moment/embeddings"
CKPT_ROOT = "/scratch/project_2019517/moment/checkpoints"
MOMENT_D_MODEL = 1024  # T5-large d_model for MOMENT-1-large

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


class MOMENTEmbeddingDataset(Dataset):
    """Loads all per-subject .npz embedding files for one split into RAM."""

    def __init__(self, embed_dir):
        npz_files = sorted(
            f for f in os.listdir(embed_dir)
            if f.endswith(".npz") and not f.startswith("_")
        )
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {embed_dir}. "
                                    "Run generate_moment_embeddings.py first.")
        emb_list, lbl_list = [], []
        for fname in npz_files:
            data = np.load(os.path.join(embed_dir, fname))
            emb_list.append(data["embeddings"].astype(np.float32))
            lbl_list.append(data["labels"].astype(np.int64))
        self.embeddings = np.concatenate(emb_list, axis=0)
        self.labels = np.concatenate(lbl_list, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx]), int(self.labels[idx])


def compute_class_weights(dataset, n_classes=5):
    counts = np.bincount(dataset.labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(head, loader, device, optimizer=None, class_weights=None):
    train = optimizer is not None
    head.train() if train else head.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = head(x)
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_base = os.path.join(EMBED_ROOT, args.modality, args.fold_key)
    out_dir = os.path.join(CKPT_ROOT, args.modality, args.fold_key)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = MOMENTEmbeddingDataset(os.path.join(embed_base, "train"))
    val_ds = MOMENTEmbeddingDataset(os.path.join(embed_base, "validation"))
    print(f"[{args.modality}] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    emb_dim = len(MODALITY_CHANNELS[args.modality]) * MOMENT_D_MODEL
    head = nn.Linear(emb_dim, 5).to(device)
    print(f"[{args.modality}] head: Linear({emb_dim}, 5)", flush=True)

    class_weights = compute_class_weights(train_ds).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    latest_path = os.path.join(out_dir, "latest.pth")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location=device)
        head.load_state_dict(checkpoint["model_state_dict"])
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
        train_loss, train_f1 = run_epoch(head, train_loader, device, optimizer, class_weights)
        val_loss, val_f1 = run_epoch(head, val_loader, device, None, class_weights)

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(head.state_dict(), os.path.join(out_dir, "best.pth"))
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            marker = " *"
        else:
            patience_counter += 1

        print(f"E{epoch + 1:03d} train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}{marker}", flush=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": head.state_dict(),
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
