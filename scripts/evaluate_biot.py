"""
Evaluate a fine-tuned BIOT checkpoint on the MESA held-out test split.

Usage:
    python scripts/evaluate_biot.py --modality EEG_ONLY
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "sleepfm"))
sys.path.insert(0, "/scratch/project_2019517/BIOT")

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from biot_dataset import BIOTSleepDataset, MODALITY_CHANNELS
from finetune_biot import build_model, SPLIT_PATH
from experiment_paths import experiment_from_checkpoint_dir, write_metrics_bundle

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=list(MODALITY_CHANNELS.keys()))
    parser.add_argument("--fold_key", default="fold_0")
    parser.add_argument("--checkpoint_dir", required=True,
                         help="A checkpoints/full_cohort/{run_name}/ directory "
                              "produced by finetune_biot.py.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp = experiment_from_checkpoint_dir(args.checkpoint_dir)
    fold_num = int(args.fold_key.replace("fold_", ""))
    fold_dir = exp.fold_dir(fold_num)
    ckpt_path = fold_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_ds = BIOTSleepDataset(SPLIT_PATH, "test", args.modality, fold_key=args.fold_key)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    print(f"[{args.modality}] test={len(test_ds)}", flush=True)

    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds, all_targets, all_paths = [], [], []
    with torch.no_grad():
        for x, y, paths in tqdm(test_loader, desc="Evaluating"):
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_targets.append(y.numpy())
            all_paths.extend(paths)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    acc = (all_preds == all_targets).mean()
    report = classification_report(
        all_targets, all_preds, labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES, zero_division=0, digits=4
    )
    print(f"\nMacro F1:  {macro_f1:.4f}\nAccuracy:  {acc:.4f}\n\n{report}")

    per_subject_rows = []
    for subject_id in sorted(set(os.path.basename(p).replace(".hdf5", "") for p in all_paths)):
        idx = [i for i, p in enumerate(all_paths) if os.path.basename(p).replace(".hdf5", "") == subject_id]
        t, p = all_targets[idx], all_preds[idx]
        per_subject_rows.append({
            "model": "biot",
            "condition": exp.modality,
            "subject_id": subject_id,
            "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
            "accuracy": round(float(accuracy_score(t, p)), 6),
            "n_valid_windows": len(idx),
        })

    config_path = fold_dir / "config.json"
    import json
    config = json.load(open(config_path)) if config_path.exists() else {}

    write_metrics_bundle(
        exp,
        metrics={
            "model": "biot", "modality": exp.modality,
            "pretrain_method": exp.pretrain_method, "split_id": exp.split_id,
            "timestamp": exp.timestamp, "overall_macro_f1": round(float(macro_f1), 6),
            "overall_accuracy": round(float(acc), 6),
        },
        classification_report_text=report,
        per_subject_rows=per_subject_rows,
        config=config,
    )
    print(f"\nWritten to {exp.results_dir}")


if __name__ == "__main__":
    main()
