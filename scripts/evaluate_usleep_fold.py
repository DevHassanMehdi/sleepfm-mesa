#!/usr/bin/env python3
"""
Compute macro F1 / per-class F1 for one U-Sleep fold's test predictions,
using the same sklearn calls as scripts/compute_metrics.py (SleepFM) so the
numbers are directly comparable.

Expects predictions written by `ut predict --majority --save_true` under
<project_dir>/predictions/test/, i.e.:
  predictions/test/<subject>_TRUE.npy
  predictions/test/majority/<subject>_PRED.npy

Usage:
    python scripts/evaluate_usleep_fold.py --fold 0 \
        --project_dir /scratch/project_2019517/usleep_mesa/projects/fold_0
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def to_class_indices(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[-1] in (5, 6):
        # one-hot / softmax volume -> argmax to class index
        arr = arr.argmax(axis=-1)
    return arr.reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--modality", default=None,
                         help="Modality label, used only for naming the output file "
                              "(predictions are read from --project_dir regardless)")
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--pred_subdir", default="predictions/test")
    parser.add_argument("--pred_folder", default="majority",
                         help="Sub-folder of pred_subdir holding *_PRED.npy "
                              "(e.g. 'majority', or a channel combo like 'EEG2+EOG-L')")
    parser.add_argument("--out_path", default=None,
                         help="Defaults to results/usleep_{modality}_fold_{fold}_results.txt "
                              "(or usleep_fold_{fold}_results.txt if --modality not given)")
    args = parser.parse_args()

    pred_dir = Path(args.project_dir) / args.pred_subdir
    true_files = sorted(pred_dir.glob("*_TRUE.npy"))
    if not true_files:
        raise SystemExit(f"No *_TRUE.npy files found in {pred_dir}")

    all_targets, all_preds = [], []
    n_missing = 0
    for true_path in true_files:
        subject_id = true_path.name[: -len("_TRUE.npy")]
        pred_path = pred_dir / args.pred_folder / f"{subject_id}_PRED.npy"
        if not pred_path.exists():
            n_missing += 1
            continue
        y_true = to_class_indices(np.load(true_path))
        y_pred = to_class_indices(np.load(pred_path))

        # UNKNOWN (class 5) has no SleepFM counterpart; drop epochs labeled UNKNOWN.
        valid = y_true < 5
        all_targets.append(y_true[valid])
        all_preds.append(y_pred[valid])

    if n_missing:
        print(f"WARNING: {n_missing}/{len(true_files)} subjects missing predictions in {args.pred_folder}")

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)

    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, target_names=CLASS_NAMES, zero_division=0)

    if args.out_path:
        out_path = Path(args.out_path)
    elif args.modality:
        out_path = Path(f"results/usleep_{args.modality}_fold_{args.fold}_results.txt")
    else:
        out_path = Path(f"results/usleep_fold_{args.fold}_results.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        label = f"{args.modality} " if args.modality else ""
        f.write(f"U-Sleep {label}fold {args.fold} ({args.pred_folder}) test results\n")
        f.write(f"n_subjects={len(true_files) - n_missing}/{len(true_files)} n_epochs={len(targets)}\n\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print(f"Fold {args.fold}: macro F1 = {macro_f1:.4f}, accuracy = {acc:.4f}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
