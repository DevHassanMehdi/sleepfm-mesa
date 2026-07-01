"""
Fine-tune LaBraM-base end-to-end on MESA PSG for 5-class sleep staging.

Loads the LaBraM-base self-supervised encoder (checkpoints/labram-base.pth),
strips the "student." prefix used by its distillation-style checkpoint, and
attaches a fresh 5-class head. Channel identity uses LaBraM's own
10-20-name-indexed positional embedding (see labram_dataset.py for the
montage mapping and its NSRR source). Only EEG_ONLY is supported -- LaBraM
is an EEG-only foundation model.

Architecture flags (qkv_bias=False, init_values=0.1) were determined
empirically by checking which configuration lets the pretrained checkpoint
load with `strict=False` and only the expected missing/unexpected keys
(fc_norm/head freshly initialized; mask_token/lm_head/norm dropped as
pretraining-only) -- not by trusting run_class_finetuning.py's CLI defaults,
which target a differently-configured run.

Usage:
    python scripts/finetune_labram.py --modality EEG_ONLY
"""
import argparse
import os
import sys
import json
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, "/scratch/project_2019517/LaBraM")

# Must be set before h5py is imported (via labram_dataset) to take effect.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from labram_dataset import LaBraMSleepDataset, MODALITY_CHANNELS, get_ch_names
from modeling_finetune import labram_base_patch200_200

SPLIT_PATH = os.path.join(REPO_ROOT, "sleepfm/configs/dataset_split_fromscratch_staging.json")
PRETRAINED_PATH = "/scratch/project_2019517/LaBraM/checkpoints/labram-base.pth"
CKPT_ROOT = "/scratch/project_2019517/labram/checkpoints"
PATCH_SIZE = 200

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# Copied from LaBraM/utils.py (not imported -- that module pulls in
# tensorboardX/pyhealth/data_processor, which we don't otherwise need).
STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
]


def get_input_chans(ch_names):
    input_chans = [0]  # slot 0 reserved for the cls token
    for ch_name in ch_names:
        input_chans.append(STANDARD_1020.index(ch_name) + 1)
    return input_chans


def build_model(modality, n_classes=5, pretrained_path=PRETRAINED_PATH):
    model = labram_base_patch200_200(
        num_classes=n_classes, in_chans=1, qkv_bias=False, init_values=0.1,
    )

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    new_dict = OrderedDict()
    for k, v in checkpoint_model.items():
        if k.startswith("student."):
            new_dict[k[len("student."):]] = v
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in new_dict and new_dict[k].shape != state_dict[k].shape:
            del new_dict[k]
    model.load_state_dict(new_dict, strict=False)
    return model


def compute_class_weights(dataset, n_classes=5):
    stages = np.array([s for _, _, s in dataset.index])
    counts = np.bincount(stages, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


N_CHUNKS = 2  # 30s epoch -> two 15s chunks of 15 one-second patches each


def forward_logits(model, x, input_chans, n_chunks=N_CHUNKS):
    """
    LaBraM's positional embedding table (time_embed) is hardcoded to 16
    one-second patches -- a 30s/200Hz epoch (30 patches) exceeds this. We
    split each epoch into n_chunks pieces (each <=16 patches, at true
    200Hz so every patch still corresponds to a real second, matching
    what the pretrained tokenizer was trained on), extract a pooled
    embedding per chunk via forward_features, average the embeddings, and
    apply the classification head once on the averaged representation.
    This preserves the full 30s of signal instead of truncating to 16s.
    """
    # LaBraM's own preprocessing convention (engine_for_finetuning.py): divide by 100.
    x = x / 100
    b, c, t = x.shape
    chunk_len = t // n_chunks
    chunk_patches = chunk_len // PATCH_SIZE
    assert chunk_patches <= 16, f"{chunk_patches} patches exceeds LaBraM's 16-patch time_embed limit"

    embeddings = []
    for i in range(n_chunks):
        chunk = x[:, :, i * chunk_len:(i + 1) * chunk_len]
        chunk = chunk.reshape(b, c, chunk_patches, PATCH_SIZE)
        embeddings.append(model.forward_features(chunk, input_chans=input_chans))
    avg_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
    return model.head(avg_embedding)


def run_epoch(model, loader, device, input_chans, optimizer=None, class_weights=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = forward_logits(model, x, input_chans)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(CKPT_ROOT, args.modality, args.fold_key)
    os.makedirs(out_dir, exist_ok=True)

    train_ds = LaBraMSleepDataset(SPLIT_PATH, "train", args.modality, fold_key=args.fold_key)
    val_ds = LaBraMSleepDataset(SPLIT_PATH, "validation", args.modality, fold_key=args.fold_key)
    print(f"[{args.modality}] train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_model(args.modality).to(device)
    input_chans = torch.tensor(get_input_chans(get_ch_names(args.modality)), device=device)
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
        train_loss, train_f1 = run_epoch(model, train_loader, device, input_chans, optimizer, class_weights)
        val_loss, val_f1 = run_epoch(model, val_loader, device, input_chans, None, class_weights)

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
