#!/usr/bin/env python3
"""
Compute metrics from saved pickle files across all folds, and write them
into the matching results/full_cohort/.../ experiment folder (metrics.json,
classification_report.txt, per_subject_results.csv, config.json).

This replaces the old pattern of manually running
    python scripts/compute_metrics.py --checkpoint_dir ... > results/SOMETHING.txt
Metrics are now written as structured files via sleepfm/experiment_paths.py,
not just printed.

Usage:
    python scripts/compute_metrics.py --checkpoint_dir /scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__EEG_ONLY__fromscratch__fold10_v1__2026-07-29_1030
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sleepfm"))
from experiment_paths import experiment_from_checkpoint_dir, write_metrics_bundle

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def per_subject_metrics(model_name, condition, all_paths, all_outputs, all_targets, all_masks):
    """Each element of all_* corresponds to one DataLoader batch. Outputs and
    targets are np arrays of shape [batch, max_seq_len, 5] and
    [batch, max_seq_len]; masks are [batch, max_seq_len] with 0=valid, 1=padding.
    """
    rows = []
    for batch_idx, paths_batch in enumerate(all_paths):
        for j in range(len(paths_batch)):
            path = paths_batch[j]
            subj_outputs = all_outputs[batch_idx][j]
            subj_targets = all_targets[batch_idx][j]
            subj_masks = all_masks[batch_idx][j]

            valid = subj_masks == 0
            t = subj_targets[valid].astype(int)
            p = np.argmax(subj_outputs[valid], axis=1)
            if len(t) == 0:
                continue

            rows.append({
                "model": model_name,
                "condition": condition,
                "subject_id": os.path.basename(path).replace(".hdf5", ""),
                "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
                "accuracy": round(float(accuracy_score(t, p)), 6),
                "n_valid_windows": int(valid.sum()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                         help="A checkpoints/full_cohort/{run_name}/ directory "
                              "(the run_name itself, not a specific fold_N).")
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n_folds', type=int, default=10,
                         help="Aggregate fold_0..fold_{n_folds-1} under checkpoint_dir. "
                              "Ignored if --fold is given.")
    parser.add_argument('--fold', type=int, default=None,
                         help="Process only this single fold index (the actual fold "
                              "number the run was trained on) instead of assuming "
                              "folds start at 0. Required when checkpoint_dir holds "
                              "just one fold_N/ (the full_cohort one-run-per-fold "
                              "naming scheme).")
    args = parser.parse_args()

    exp = experiment_from_checkpoint_dir(args.checkpoint_dir)
    base = Path(args.checkpoint_dir)

    fold_f1s = []
    all_targets_agg = []
    all_preds_agg = []
    per_subject_rows = []
    folds_used = 0

    folds_to_process = [args.fold] if args.fold is not None else range(args.n_folds)
    for fold in folds_to_process:
        fold_dir = base / f'fold_{fold}'
        try:
            with open(fold_dir / f'{args.split}_all_outputs.pickle', 'rb') as f:
                outputs = pickle.load(f)
            with open(fold_dir / f'{args.split}_all_targets.pickle', 'rb') as f:
                targets = pickle.load(f)
            with open(fold_dir / f'{args.split}_all_masks.pickle', 'rb') as f:
                masks = pickle.load(f)
            with open(fold_dir / f'{args.split}_all_paths.pickle', 'rb') as f:
                paths = pickle.load(f)
        except FileNotFoundError:
            print(f'Fold {fold}: missing pickle files, skipping')
            continue

        folds_used += 1
        preds_flat = np.concatenate([o.reshape(-1, 5) for o in outputs], axis=0)
        targets_flat = np.concatenate([t.reshape(-1) for t in targets], axis=0)
        masks_flat = np.concatenate([m.reshape(-1) for m in masks], axis=0)

        valid = masks_flat == 0
        t = targets_flat[valid].astype(int)
        p = np.argmax(preds_flat[valid], axis=1)

        fold_f1 = f1_score(t, p, average='macro', zero_division=0)
        fold_f1s.append(fold_f1)
        print(f'Fold {fold}: macro F1 = {fold_f1:.4f}, n={len(t)}')
        all_targets_agg.extend(t)
        all_preds_agg.extend(p)

        per_subject_rows.extend(
            per_subject_metrics(exp.model, exp.modality, paths, outputs, targets, masks)
        )

    print()
    print(f'Mean macro F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}')

    all_targets_agg = np.array(all_targets_agg)
    all_preds_agg = np.array(all_preds_agg)
    overall_macro_f1 = f1_score(all_targets_agg, all_preds_agg, average="macro", zero_division=0)
    overall_accuracy = accuracy_score(all_targets_agg, all_preds_agg)
    report_text = classification_report(
        all_targets_agg, all_preds_agg,
        target_names=STAGE_NAMES, zero_division=0, digits=4)

    print('=' * 50)
    print(f'OVERALL ({args.split} set, {folds_used} folds)')
    print('=' * 50)
    print(f'Macro F1:  {overall_macro_f1:.4f}')
    print(f'Accuracy:  {overall_accuracy:.4f}')
    print()
    print(report_text)

    metrics = {
        "model": exp.model,
        "modality": exp.modality,
        "pretrain_method": exp.pretrain_method,
        "split_id": exp.split_id,
        "timestamp": exp.timestamp,
        "split": args.split,
        "n_folds_used": folds_used,
        "per_fold_macro_f1": [round(float(f1), 6) for f1 in fold_f1s],
        "mean_macro_f1": round(float(np.mean(fold_f1s)), 6),
        "std_macro_f1": round(float(np.std(fold_f1s)), 6),
        "overall_macro_f1": round(float(overall_macro_f1), 6),
        "overall_accuracy": round(float(overall_accuracy), 6),
    }

    config_path = base / "config.json"
    config = json.load(open(config_path)) if config_path.exists() else {}

    write_metrics_bundle(
        exp,
        metrics=metrics,
        classification_report_text=report_text,
        per_subject_rows=per_subject_rows,
        config=config,
    )
    print(f"\nWritten to {exp.results_dir}")


if __name__ == '__main__':
    main()
