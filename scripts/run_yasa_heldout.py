"""
Run YASA sleep staging on the MESA held-out test split for a given modality.

YASA is an EEG-anchored model. ECG_ONLY is not supported (skipped at
submission level). EEG_ECG is run with EEG1 only — ECG is ignored by YASA
since SleepStaging accepts no ECG argument.

Usage:
    python scripts/run_yasa_heldout.py --modality EEG_ONLY
    python scripts/run_yasa_heldout.py --modality EEG_ECG
"""
import argparse
import json
import os

import h5py
import mne
import numpy as np
import pandas as pd
import yasa
from sklearn.metrics import classification_report, f1_score

mne.set_log_level("WARNING")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATH = os.path.join(REPO_ROOT, "sleepfm/configs/dataset_split_fromscratch_staging.json")
HDF5_DIR   = os.path.join(REPO_ROOT, "data/mesa/hdf5")
LABELS_DIR = os.path.join(REPO_ROOT, "data/mesa/labels")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

SF         = 128       # MESA native sampling frequency
EPOCH_SEC  = 30
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# YASA only uses EEG. For EEG_ECG we run YASA on EEG1 only and note this
# clearly in the result header so comparisons are unambiguous.
MODALITY_EEG = {
    "EEG_ONLY": "EEG1",
    "EEG_ECG":  "EEG1",   # ECG ignored by YASA
}

MODALITY_NOTE = {
    "EEG_ONLY": "EEG1 only (standard YASA configuration)",
    "EEG_ECG":  "EEG1 only — YASA has no ECG input; ECG channel ignored",
}


def stage_subject(hdf5_path, eeg_name):
    """Return (pred_int_array, n_yasa_epochs) for one subject.

    Loads the full EEG1 signal, builds an MNE Raw object at 128 Hz,
    and runs YASA's default LightGBM classifier.
    """
    with h5py.File(hdf5_path, "r") as hf:
        eeg = hf[eeg_name][:].astype(np.float64)

    info = mne.create_info([eeg_name], sfreq=SF, ch_types=["eeg"])
    raw  = mne.io.RawArray(eeg[np.newaxis, :], info, verbose=False)

    sls   = yasa.SleepStaging(raw, eeg_name=eeg_name)
    hypno = sls.predict()            # yasa.Hypnogram, 5-stage
    pred  = hypno.as_int().values.astype(int)  # WAKE=0,N1=1,N2=2,N3=3,REM=4
    return pred


def load_labels(label_path):
    """Return (epoch_indices, stage_ints) for valid (0-4) epochs."""
    df = pd.read_csv(label_path)
    valid = df["StageNumber"].isin([0, 1, 2, 3, 4])
    df = df[valid].reset_index(drop=True)
    epoch_idx = (df["Start"].values / EPOCH_SEC).astype(int)
    stages    = df["StageNumber"].values.astype(int)
    return epoch_idx, stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True,
                        choices=list(MODALITY_EEG.keys()))
    parser.add_argument("--fold_key", default="fold_0")
    args = parser.parse_args()

    eeg_name = MODALITY_EEG[args.modality]

    with open(SPLIT_PATH) as f:
        test_files = json.load(f)[args.fold_key]["test"]

    all_true, all_pred = [], []
    n_skipped = 0

    for i, fname in enumerate(test_files):
        sid        = fname.replace(".hdf5", "")
        hdf5_path  = os.path.join(HDF5_DIR, fname)
        label_path = os.path.join(LABELS_DIR, f"{sid}.csv")

        if not os.path.exists(hdf5_path) or not os.path.exists(label_path):
            print(f"[{i+1}/{len(test_files)}] SKIP {sid}: file missing", flush=True)
            n_skipped += 1
            continue

        if eeg_name not in h5py.File(hdf5_path, "r").keys():
            print(f"[{i+1}/{len(test_files)}] SKIP {sid}: {eeg_name} not in HDF5", flush=True)
            n_skipped += 1
            continue

        try:
            pred_all  = stage_subject(hdf5_path, eeg_name)
            epoch_idx, true = load_labels(label_path)

            # Clip to epochs YASA actually predicted
            valid = epoch_idx < len(pred_all)
            epoch_idx = epoch_idx[valid]
            true      = true[valid]
            pred      = pred_all[epoch_idx]

            all_true.append(true)
            all_pred.append(pred)
            f1 = f1_score(true, pred, average="macro", zero_division=0)
            print(f"[{i+1}/{len(test_files)}] {sid}: macro_f1={f1:.4f} "
                  f"epochs={len(true)}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(test_files)}] ERROR {sid}: {e}", flush=True)
            n_skipped += 1

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    acc      = (all_pred == all_true).mean()
    report   = classification_report(
        all_true, all_pred, labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES, zero_division=0, digits=4,
    )

    lines = [
        f"YASA ({args.modality}) — MESA held-out test split",
        "=" * 50,
        f"EEG channel: {eeg_name}",
        f"Note: {MODALITY_NOTE[args.modality]}",
        f"Subjects evaluated: {len(test_files) - n_skipped}  (skipped: {n_skipped})",
        f"Total epochs: {len(all_true)}",
        "",
        f"Macro F1:  {macro_f1:.4f}",
        f"Accuracy:  {acc:.4f}",
        "",
        report,
    ]
    result_text = "\n".join(lines)
    print("\n" + result_text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"yasa_{args.modality}_heldout_results.txt")
    with open(out_path, "w") as f:
        f.write(result_text)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
