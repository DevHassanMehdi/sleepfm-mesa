#!/usr/bin/env python3
"""
Aggregate per-fold U-Sleep results (results/usleep_fold_{N}_results.txt,
written by evaluate_usleep_fold.py) into a single summary, mirroring
scripts/compute_metrics.py's output format for direct comparison with
SleepFM's results.

Usage:
    python scripts/compute_usleep_metrics.py --n_folds 10
"""
import argparse
import re
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--modality", default=None,
                         help="If given, aggregates usleep_{modality}_fold_{N}_results.txt "
                              "instead of usleep_fold_{N}_results.txt")
    parser.add_argument("--out_path", default=None,
                         help="Defaults to results/usleep_{modality}_results.txt "
                              "(or results/usleep_baseline_results.txt if --modality not given)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fold_f1s = []
    lines_out = []
    file_prefix = f"usleep_{args.modality}_fold_" if args.modality else "usleep_fold_"

    for fold in range(args.n_folds):
        fold_path = results_dir / f"{file_prefix}{fold}_results.txt"
        if not fold_path.exists():
            print(f"Fold {fold}: missing {fold_path}, skipping")
            continue
        text = fold_path.read_text()
        m = re.search(r"Macro F1:\s*([0-9.]+)", text)
        if not m:
            print(f"Fold {fold}: could not parse macro F1 from {fold_path}, skipping")
            continue
        f1 = float(m.group(1))
        fold_f1s.append(f1)
        print(f"Fold {fold}: macro F1 = {f1:.4f}")

    if not fold_f1s:
        raise SystemExit("No fold results found.")

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    modality_label = f" ({args.modality})" if args.modality else ""
    lines_out.append(f"U-Sleep baseline{modality_label} - MESA 10-fold CV (matching SleepFM split)")
    lines_out.append(f"Folds evaluated: {len(fold_f1s)}/{args.n_folds}")
    lines_out.append("")
    for fold, f1 in zip(range(args.n_folds), fold_f1s):
        lines_out.append(f"Fold {fold}: macro F1 = {f1:.4f}")
    lines_out.append("")
    lines_out.append(f"Mean macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")

    if args.out_path:
        out_path = Path(args.out_path)
    elif args.modality:
        out_path = Path(f"results/usleep_{args.modality}_results.txt")
    else:
        out_path = Path("results/usleep_baseline_results.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out) + "\n")

    print()
    print(f"Mean macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
