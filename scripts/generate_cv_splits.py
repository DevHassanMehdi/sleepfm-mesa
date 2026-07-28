#!/usr/bin/env python3
"""
Generate a subject-level N-fold CV split over the full MESA cohort, for
the new results/full_cohort/ + checkpoints/full_cohort/ pipeline.

Subject-level guarantee: every subject ID is assigned to exactly one of N
partitions once (a single random.shuffle + even split of the full subject
list), and that partition assignment is used unchanged for every fold's
train/validation/test role -- so all of a subject's epochs always move
together, never split across train/val/test within a single fold, and
never appear in more than one role across the N rotations.

Rotation scheme (same convention as download_mesa.py's existing
generate_10fold_split, just parameterized by N): for fold i,
    test       = partition[i]
    validation = partition[(i+1) % N]
    train      = all other N-2 partitions

No stratification: MESA's locally available metadata is limited to
per-epoch sleep-stage labels (Start/Stop/StageName/StageNumber) -- there
is no AHI/age/sex/demographic file present anywhere in this project's
data directories, so stratifying by any of those would require a
separate NSRR download of MESA's clinical/demographics dataset first.
Plain random subject-level assignment is used instead (permitted
explicitly for this case).

Usage:
    python scripts/generate_cv_splits.py --n_folds 5  --split_id fold5_v1
    python scripts/generate_cv_splits.py --n_folds 10 --split_id fold10_v1
"""
import argparse
import json
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HDF5_FULL_DIR = Path("/scratch/project_2019517/sleepfm-data/mesa/hdf5_full")
LABELS_DIR = Path("/scratch/project_2019517/sleepfm-data/mesa/labels")


def get_subject_pool():
    """Every subject with both a raw HDF5 and a label CSV -- the full
    usable cohort (confirmed 1944/1944 as of 2026-07-28)."""
    hdf5_ids = {f[:-len(".hdf5")] for f in os.listdir(HDF5_FULL_DIR) if f.endswith(".hdf5")}
    label_ids = {f[:-len(".csv")] for f in os.listdir(LABELS_DIR) if f.endswith(".csv")}
    return sorted(hdf5_ids & label_ids)


def generate_split(subject_ids, n_folds, seed=42):
    ids = list(subject_ids)
    random.Random(seed).shuffle(ids)

    n = len(ids)
    fold_size = n // n_folds
    partitions = [ids[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    # any remainder subjects go into the last partition, same as
    # download_mesa.py's existing generate_10fold_split
    partitions[-1].extend(ids[n_folds * fold_size:])

    split = {}
    for i in range(n_folds):
        test_ids = partitions[i]
        val_ids = partitions[(i + 1) % n_folds]
        train_ids = []
        for j in range(n_folds):
            if j != i and j != (i + 1) % n_folds:
                train_ids.extend(partitions[j])
        split[f"fold_{i}"] = {
            "train": [f"{sid}.hdf5" for sid in train_ids],
            "validation": [f"{sid}.hdf5" for sid in val_ids],
            "test": [f"{sid}.hdf5" for sid in test_ids],
        }
    return split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, required=True)
    parser.add_argument("--split_id", type=str, required=True,
                         help='e.g. "fold5_v1" or "fold10_v1"')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subject_ids = get_subject_pool()
    print(f"Subject pool: {len(subject_ids)} (hdf5_full ∩ labels)")

    split = generate_split(subject_ids, args.n_folds, seed=args.seed)

    for fold_key, fold in split.items():
        total = len(fold["train"]) + len(fold["validation"]) + len(fold["test"])
        assert total == len(subject_ids), f"{fold_key}: {total} != {len(subject_ids)}"
        assert not (set(fold["train"]) & set(fold["validation"]))
        assert not (set(fold["train"]) & set(fold["test"]))
        assert not (set(fold["validation"]) & set(fold["test"]))
        print(f"  {fold_key}: train={len(fold['train'])} "
              f"validation={len(fold['validation'])} test={len(fold['test'])}")

    out_path = REPO_ROOT / "sleepfm" / "configs" / f"dataset_split_{args.split_id}.json"
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
