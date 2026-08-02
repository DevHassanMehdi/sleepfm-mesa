"""
Evaluate a trained SensorLM checkpoint on the MESA held-out test split.

Usage:
    python scripts/evaluate_sensorlm.py --modality EEG_ONLY
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "sleepfm"))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from sensorlm_dataset import MODALITY_CHANNELS, SensorLMSleepDataset, HDF5_DIR
from sensorlm_model import SensorLMEncoder
from experiment_paths import experiment_from_checkpoint_dir, write_metrics_bundle

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True,
                        choices=list(MODALITY_CHANNELS.keys()))
    parser.add_argument("--fold_key",    default="fold_0")
    parser.add_argument("--checkpoint_dir", required=True,
                         help="A checkpoints/full_cohort/{run_name}/ directory "
                              "produced by finetune_sensorlm.py.")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hdf5_dir", type=str, default=HDF5_DIR,
                         help="Override the MESA HDF5 directory (default: "
                              "sensorlm_dataset.py's built-in 350-subject "
                              "Puhti-era path). Pass the full-cohort path "
                              "for full-cohort evaluation -- must match "
                              "whatever --hdf5_dir the checkpoint was "
                              "fine-tuned with.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = experiment_from_checkpoint_dir(args.checkpoint_dir)
    fold_num = int(args.fold_key.replace("fold_", ""))
    fold_dir = exp.fold_dir(fold_num)
    ckpt_path = fold_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Derived from exp.split_id (parsed from checkpoint_dir's own name) so
    # eval always uses the exact split file that produced this checkpoint --
    # not a separately-passed, possibly-mismatched --split_id.
    split_path = os.path.join(REPO_ROOT, f"sleepfm/configs/dataset_split_{exp.split_id}.json")
    test_ds = SensorLMSleepDataset(split_path, "test", args.modality, args.fold_key,
                                    hdf5_dir=args.hdf5_dir)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"[{args.modality}] test={len(test_ds)}", flush=True)

    n_channels = len(MODALITY_CHANNELS[args.modality])
    model      = SensorLMEncoder(n_channels=n_channels).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds, all_targets, all_paths = [], [], []
    with torch.no_grad():
        for x, y, paths in tqdm(loader, desc="Evaluating"):
            preds = model(x.to(device)).argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())
            all_paths.extend(paths)

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    acc      = (all_preds == all_targets).mean()
    report   = classification_report(
        all_targets, all_preds, labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES, zero_division=0, digits=4,
    )
    print(f"\nMacro F1:  {macro_f1:.4f}\nAccuracy:  {acc:.4f}\n\n{report}")

    per_subject_rows = []
    for subject_id in sorted(set(os.path.basename(p).replace(".hdf5", "") for p in all_paths)):
        idx = [i for i, p in enumerate(all_paths) if os.path.basename(p).replace(".hdf5", "") == subject_id]
        t, p = all_targets[idx], all_preds[idx]
        per_subject_rows.append({
            "model": "sensorlm",
            "condition": exp.modality,
            "subject_id": subject_id,
            "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
            "accuracy": round(float(accuracy_score(t, p)), 6),
            "n_valid_windows": len(idx),
        })

    config_path = fold_dir / "config.json"
    config = json.load(open(config_path)) if config_path.exists() else {}

    write_metrics_bundle(
        exp,
        fold_num,
        metrics={
            "model": "sensorlm", "modality": exp.modality,
            "pretrain_method": exp.pretrain_method, "split_id": exp.split_id,
            "timestamp": exp.timestamp, "fold": fold_num,
            "overall_macro_f1": round(float(macro_f1), 6),
            "overall_accuracy": round(float(acc), 6),
        },
        classification_report_text=report,
        per_subject_rows=per_subject_rows,
        config=config,
    )
    print(f"\nWritten to {exp.results_dir / f'fold_{fold_num}'}", flush=True)


if __name__ == "__main__":
    main()
