#!/usr/bin/env python3
"""
Compute metrics from saved pickle files across all folds.

Usage:
    python scripts/compute_metrics.py --checkpoint_dir sleepfm/checkpoints/model_base/SleepEventLSTMClassifier_mesa_labels_BAS
"""
import argparse
import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n_folds', type=int, default=10)
    args = parser.parse_args()

    base = Path(args.checkpoint_dir)
    fold_f1s = []
    all_targets = []
    all_preds = []

    for fold in range(args.n_folds):
        fold_dir = base / f'fold_{fold}' / 'mesa' / args.split
        try:
            with open(fold_dir / 'all_outputs.pickle', 'rb') as f:
                outputs = pickle.load(f)
            with open(fold_dir / 'all_targets.pickle', 'rb') as f:
                targets = pickle.load(f)
            with open(fold_dir / 'all_masks.pickle', 'rb') as f:
                masks = pickle.load(f)
        except FileNotFoundError:
            print(f'Fold {fold}: missing pickle files, skipping')
            continue

        preds_flat = np.concatenate([o.reshape(-1, 5) for o in outputs], axis=0)
        targets_flat = np.concatenate([t.reshape(-1) for t in targets], axis=0)
        masks_flat = np.concatenate([m.reshape(-1) for m in masks], axis=0)

        valid = masks_flat == 0
        t = targets_flat[valid].astype(int)
        p = np.argmax(preds_flat[valid], axis=1)

        fold_f1 = f1_score(t, p, average='macro', zero_division=0)
        fold_f1s.append(fold_f1)
        print(f'Fold {fold}: macro F1 = {fold_f1:.4f}, n={len(t)}')
        all_targets.extend(t)
        all_preds.extend(p)

    print()
    print(f'Mean macro F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}')
    print()
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    print('='*50)
    print(f'OVERALL ({args.split} set, {len(fold_f1s)} folds)')
    print('='*50)
    print(f'Macro F1:  {f1_score(all_targets, all_preds, average="macro", zero_division=0):.4f}')
    print(f'Accuracy:  {accuracy_score(all_targets, all_preds):.4f}')
    print()
    print(classification_report(all_targets, all_preds,
        target_names=['Wake', 'N1', 'N2', 'N3', 'REM'], zero_division=0))


if __name__ == '__main__':
    main()
