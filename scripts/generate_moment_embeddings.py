"""
Pre-compute MOMENT frozen-encoder embeddings for all MESA subjects.

Runs the T5 encoder once per 30s epoch (no gradient computation) and saves
per-subject .npz files. These cached files let finetune_moment_cached.py
train only the classification head — epochs take minutes instead of hours.

File layout (per-subject to allow resume after interruption):
  {EMBED_ROOT}/{MODALITY}/{fold_key}/{split}/{subject_id}.npz
    'embeddings': float32 (n_epochs, emb_dim)
    'labels':     int64   (n_epochs,)

The embedding captured is the input to model.head.linear — the concatenated
per-channel T5 representation of shape (n_channels * 1024,) — identical to
what MOMENT's own ClassificationHead trains on.

Usage:
    python scripts/generate_moment_embeddings.py --modality EEG_ONLY
"""
import argparse
import os
import sys
import json
from collections import defaultdict
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from moment_dataset import MESADataset, MODALITY_CHANNELS
from momentfm import MOMENTPipeline

SPLIT_PATH = os.path.join(REPO_ROOT, "sleepfm/configs/dataset_split_fromscratch_staging.json")
PRETRAINED_NAME = "AutonLab/MOMENT-1-large"
EMBED_ROOT = "/scratch/project_2019517/moment/embeddings"


def build_frozen_encoder(modality, pretrained_name=PRETRAINED_NAME):
    n_channels = len(MODALITY_CHANNELS[modality])
    model = MOMENTPipeline.from_pretrained(
        pretrained_name,
        model_kwargs={
            "task_name": "classification",
            "n_channels": n_channels,
            "num_class": 5,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "reduction": "concat",
        },
    )
    model.init()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def extract_embeddings(model, loader, device):
    """Run loader through frozen encoder; return (embeddings, labels) as numpy arrays.

    Hooks into model.head.linear to capture the pre-classification feature
    vector (shape: batch_size x emb_dim) for each batch.
    """
    captured = []

    def _hook(module, inp, out):
        captured.append(inp[0].detach().cpu())

    handle = model.head.linear.register_forward_hook(_hook)
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mask = torch.ones((x.shape[0], x.shape[-1]), device=device)
            model(x_enc=x, input_mask=mask)
            all_labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))

    handle.remove()

    embeddings = torch.cat(captured, dim=0).numpy().astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int64)
    return embeddings, labels


def process_split(model, ds, split_out_dir, args, device):
    done_marker = os.path.join(split_out_dir, "_done")
    if os.path.exists(done_marker):
        print(f"  already complete — skipping", flush=True)
        return

    os.makedirs(split_out_dir, exist_ok=True)

    # Group flat index by subject (all epochs for a subject share the same hdf5_path)
    subject_indices = defaultdict(list)
    for i, (hdf5_path, _, _) in enumerate(ds.index):
        subject_id = os.path.basename(hdf5_path).replace(".hdf5", "")
        subject_indices[subject_id].append(i)

    n_subjects = len(subject_indices)
    print(f"  {n_subjects} subjects, {len(ds)} epochs", flush=True)

    done_count = 0
    skip_count = 0
    for subject_id, indices in subject_indices.items():
        out_path = os.path.join(split_out_dir, f"{subject_id}.npz")
        if os.path.exists(out_path):
            done_count += 1
            continue

        # Guard against subjects with missing channels (e.g. missing Therm for 3 RESP subjects)
        hdf5_path = ds.index[indices[0]][0]
        with h5py.File(hdf5_path, "r") as hf:
            missing = [ch for ch in ds.channels if ch not in hf]
        if missing:
            print(f"  WARNING: skipping {subject_id} — missing channel(s) {missing}", flush=True)
            skip_count += 1
            continue

        subset = Subset(ds, indices)
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

        embs, labels = extract_embeddings(model, loader, device)
        np.savez(out_path, embeddings=embs, labels=labels)
        done_count += 1

        if done_count % 20 == 0:
            print(f"  {done_count}/{n_subjects} subjects", flush=True)

    open(done_marker, "w").close()
    print(f"  complete — {done_count} saved, {skip_count} skipped", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=list(MODALITY_CHANNELS.keys()))
    parser.add_argument("--fold_key", default="fold_0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.modality}] device={device}", flush=True)

    model = build_frozen_encoder(args.modality).to(device)
    n_channels = len(MODALITY_CHANNELS[args.modality])
    emb_dim = n_channels * 1024  # MOMENT-1-large d_model = 1024
    print(f"[{args.modality}] encoder loaded  emb_dim={emb_dim}", flush=True)

    embed_base = os.path.join(EMBED_ROOT, args.modality, args.fold_key)

    for split in ("train", "validation", "test"):
        print(f"[{args.modality}] split={split}", flush=True)
        ds = MESADataset(SPLIT_PATH, split, args.modality, fold_key=args.fold_key)
        split_dir = os.path.join(embed_base, split)
        process_split(model, ds, split_dir, args, device)

    print(f"Done. Embeddings at {embed_base}", flush=True)


if __name__ == "__main__":
    main()
