"""
Evaluate the cached-embedding MOMENT head on the MESA held-out test split.

Loads pre-computed test embeddings from:
  /scratch/project_2019517/moment/embeddings/{MODALITY}/{fold_key}/test/*.npz

Loads the best head checkpoint from:
  /scratch/project_2019517/moment/checkpoints/{MODALITY}/{fold_key}/best.pth

Writes results to:
  results/moment_cached_{MODALITY}_results.txt

Usage:
    python scripts/evaluate_moment_cached.py --modality EEG_ONLY
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from moment_dataset import MODALITY_CHANNELS
from finetune_moment_cached import MOMENTEmbeddingDataset, EMBED_ROOT, CKPT_ROOT, MOMENT_D_MODEL

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=list(MODALITY_CHANNELS.keys()))
    parser.add_argument("--fold_key", default="fold_0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(CKPT_ROOT, args.modality, args.fold_key, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    embed_dir = os.path.join(EMBED_ROOT, args.modality, args.fold_key, "test")
    test_ds = MOMENTEmbeddingDataset(embed_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    print(f"[{args.modality}] test={len(test_ds)}", flush=True)

    emb_dim = len(MODALITY_CHANNELS[args.modality]) * MOMENT_D_MODEL
    head = nn.Linear(emb_dim, 5).to(device)
    head.load_state_dict(torch.load(ckpt_path, map_location=device))
    head.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = head(x)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_targets.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    acc = (all_preds == all_targets).mean()
    report = classification_report(
        all_targets, all_preds, labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES, zero_division=0,
        digits=4,
    )

    lines = [
        f"MOMENT CACHED ({args.modality}) — MESA held-out test split",
        "=" * 50,
        f"Channels: {', '.join(MODALITY_CHANNELS[args.modality])}",
        "Pretrained: AutonLab/MOMENT-1-large (encoder frozen, linear head trained on cached embeddings)",
        "",
        f"Macro F1:  {macro_f1:.4f}",
        f"Accuracy:  {acc:.4f}",
        "",
        report,
    ]
    result_text = "\n".join(lines)
    print("\n" + result_text)

    out_path = os.path.join(REPO_ROOT, "results", f"moment_cached_{args.modality}_results.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(result_text)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
