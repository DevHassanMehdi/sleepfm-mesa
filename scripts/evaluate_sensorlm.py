"""
Evaluate a trained SensorLM checkpoint on the MESA held-out test split.

Usage:
    python scripts/evaluate_sensorlm.py --modality EEG_ONLY
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from sensorlm_dataset import MODALITY_CHANNELS, SensorLMSleepDataset
from sensorlm_model import SensorLMEncoder
from finetune_sensorlm import CKPT_ROOT, SPLIT_PATH

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True,
                        choices=list(MODALITY_CHANNELS.keys()))
    parser.add_argument("--fold_key",    default="fold_0")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(CKPT_ROOT, args.modality, args.fold_key, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_ds = SensorLMSleepDataset(SPLIT_PATH, "test", args.modality, args.fold_key)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"[{args.modality}] test={len(test_ds)}", flush=True)

    n_channels = len(MODALITY_CHANNELS[args.modality])
    model      = SensorLMEncoder(n_channels=n_channels).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            preds = model(x.to(device)).argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    acc      = (all_preds == all_targets).mean()
    report   = classification_report(
        all_targets, all_preds, labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES, zero_division=0, digits=4,
    )

    lines = [
        f"SensorLM FROM SCRATCH ({args.modality}) — MESA held-out test split",
        "=" * 55,
        f"Channels: {', '.join(MODALITY_CHANNELS[args.modality])}",
        "Architecture: ViT-B, patch=(64t,1ch), MAP pool",
        "Trained from random init on MESA train split",
        "",
        f"Macro F1:  {macro_f1:.4f}",
        f"Accuracy:  {acc:.4f}",
        "",
        report,
    ]
    result_text = "\n".join(lines)
    print("\n" + result_text)

    out_path = os.path.join(REPO_ROOT, "results",
                            f"sensorlm_{args.modality}_results.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(result_text)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
