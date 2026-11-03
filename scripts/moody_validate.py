"""Validate converted Moody/PhysioNet 2026 HDF5 + label CSV pairs.

For each subject, checks:
  - HDF5 keys are exactly {EEG1, EEG2, EEG3, EKG}
  - signal epochs (n_samples // 128 // 30) == label CSV row count
  - no NaN or inf in any of the 4 channel arrays
  - no channel is flat/all-zero (std == 0)

Prints one line per failure (never a full subject ID list on success,
to keep output short); exits 1 if any record fails.
"""
import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd

os.umask(0o077)

EXPECTED_KEYS = {"EEG1", "EEG2", "EEG3", "EKG"}
SAMPLE_RATE = 128
EPOCH_SEC = 30


def validate_one(hdf5_path, csv_path):
    problems = []
    with h5py.File(hdf5_path, "r") as hf:
        keys = set(hf.keys())
        if keys != EXPECTED_KEYS:
            problems.append(f"HDF5 keys {keys} != {EXPECTED_KEYS}")
            return problems  # can't meaningfully check further

        arrays = {k: hf[k][:] for k in EXPECTED_KEYS}

    n_samples = len(next(iter(arrays.values())))
    sig_epochs = n_samples // (SAMPLE_RATE * EPOCH_SEC)

    df = pd.read_csv(csv_path)
    label_rows = len(df)
    if sig_epochs != label_rows:
        problems.append(f"signal epochs {sig_epochs} != label rows {label_rows}")

    for ch, arr in arrays.items():
        arr64 = arr.astype(np.float64)
        if np.isnan(arr64).any():
            problems.append(f"{ch}: contains NaN")
        if np.isinf(arr64).any():
            problems.append(f"{ch}: contains inf")
        if np.std(arr64) == 0:
            problems.append(f"{ch}: flat/all-zero (std=0)")

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir", type=str, required=True)
    parser.add_argument("--labels_dir", type=str, required=True)
    args = parser.parse_args()

    hdf5_files = sorted(glob.glob(os.path.join(args.hdf5_dir, "*.hdf5")))
    n_ok, n_failed, n_missing_label = 0, 0, 0
    failures = []

    for hdf5_path in hdf5_files:
        subject_id = os.path.basename(hdf5_path).replace(".hdf5", "")
        csv_path = os.path.join(args.labels_dir, f"{subject_id}.csv")
        if not os.path.exists(csv_path):
            print(f"FAIL: {subject_id}: missing label CSV", flush=True)
            n_missing_label += 1
            continue
        problems = validate_one(hdf5_path, csv_path)
        if problems:
            for p in problems:
                print(f"FAIL: {subject_id}: {p}", flush=True)
            n_failed += 1
            failures.append(subject_id)
        else:
            n_ok += 1

    print(f"\nValidation done. ok={n_ok} failed={n_failed} "
          f"missing_label={n_missing_label} total={len(hdf5_files)}", flush=True)
    if failures or n_missing_label:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
