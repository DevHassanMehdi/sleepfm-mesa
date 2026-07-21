"""
Run YASA sleep staging on the MESA held-out test split for a given modality.

YASA is an EEG-anchored model. ECG_ONLY is not supported (skipped at
submission level). EEG_ECG is run with EEG1 only — ECG is ignored by YASA
since SleepStaging accepts no ECG argument.

Usage:
    python scripts/run_yasa_heldout.py --modality EEG_ONLY
    python scripts/run_yasa_heldout.py --modality EEG_ECG

Per-subject CSV output
----------------------
Saves results/per_subject/yasa_{modality}_per_subject.csv with columns:
  model, condition, subject_id, macro_f1, accuracy, n_epochs

When --modality EEG_ONLY is run, also writes yasa_eeg_ecg_per_subject.csv
as a deliberate duplicate (condition column changed to EEG_ECG).  YASA
produces identical predictions for EEG_ONLY and EEG_ECG because it has no
ECG input — the two CSVs are bit-for-bit identical except the condition
column.  This duplication is intentional so downstream code can iterate all
model × condition pairs without special-casing YASA.

IMPORTANT — subject-set mismatch
---------------------------------
YASA's per-subject CSVs contain 49 subjects, NOT 50.  Subject
mesa-sleep-0555 is permanently absent from YASA's output because its HDF5
file is missing from disk.  All other baseline models (BIOT, LaBraM, MOMENT,
SensorLM) include this subject (50 rows, 60,705 total epochs each).  Any
join between YASA and the other per-subject CSVs on subject_id will silently
drop mesa-sleep-0555 for YASA; be aware of this asymmetry.
"""
import argparse
import csv
import datetime
import json
import os
import sys

import h5py
import mne
import numpy as np
import pandas as pd
import yasa
from sklearn.metrics import accuracy_score, classification_report, f1_score

mne.set_log_level("WARNING")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATH      = os.path.join(REPO_ROOT, "sleepfm/configs/dataset_split_fromscratch_staging.json")
HDF5_DIR        = os.path.join(REPO_ROOT, "data/mesa/hdf5")
LABELS_DIR      = os.path.join(REPO_ROOT, "data/mesa/labels")
RESULTS_DIR     = os.path.join(REPO_ROOT, "results")
PER_SUBJECT_DIR = os.path.join(REPO_ROOT, "results", "per_subject")

# Provenance: aggregate macro F1 from the original YASA runs (jobs 35351012/35351013)
EXPECTED_AGG_F1 = 0.4720
TOLERANCE       = 1e-3

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
    subject_rows = []
    skipped_sids = []
    n_skipped = 0

    for i, fname in enumerate(test_files):
        sid        = fname.replace(".hdf5", "")
        hdf5_path  = os.path.join(HDF5_DIR, fname)
        label_path = os.path.join(LABELS_DIR, f"{sid}.csv")

        if not os.path.exists(hdf5_path) or not os.path.exists(label_path):
            print(f"[{i+1}/{len(test_files)}] SKIP {sid}: file missing", flush=True)
            skipped_sids.append(sid)
            n_skipped += 1
            continue

        if eeg_name not in h5py.File(hdf5_path, "r").keys():
            print(f"[{i+1}/{len(test_files)}] SKIP {sid}: {eeg_name} not in HDF5", flush=True)
            skipped_sids.append(sid)
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

            subj_f1  = f1_score(true, pred, average="macro", zero_division=0)
            subj_acc = accuracy_score(true, pred)
            subject_rows.append({
                "model":      "YASA",
                "condition":  args.modality,
                "subject_id": sid,
                "macro_f1":   round(float(subj_f1),  6),
                "accuracy":   round(float(subj_acc), 6),
                "n_epochs":   len(true),
            })
            print(f"[{i+1}/{len(test_files)}] {sid}: macro_f1={subj_f1:.4f} "
                  f"acc={subj_acc:.4f} epochs={len(true)}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(test_files)}] ERROR {sid}: {e}", flush=True)
            skipped_sids.append(sid)
            n_skipped += 1

    all_true_arr = np.concatenate(all_true)
    all_pred_arr = np.concatenate(all_pred)

    macro_f1 = f1_score(all_true_arr, all_pred_arr, average="macro", zero_division=0)
    acc      = accuracy_score(all_true_arr, all_pred_arr)
    report   = classification_report(
        all_true_arr, all_pred_arr, labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES, zero_division=0, digits=4,
    )

    lines = [
        f"YASA ({args.modality}) — MESA held-out test split",
        "=" * 50,
        f"EEG channel: {eeg_name}",
        f"Note: {MODALITY_NOTE[args.modality]}",
        f"Subjects evaluated: {len(test_files) - n_skipped}  (skipped: {n_skipped})",
        f"Skipped subject IDs: {skipped_sids}",
        f"Total epochs: {len(all_true_arr)}",
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

    # ── Provenance check ────────────────────────────────────────────────────
    match = abs(macro_f1 - EXPECTED_AGG_F1) <= TOLERANCE
    status = "MATCH" if match else "MISMATCH"
    print(f"\nProvenance: aggregate macro F1={macro_f1:.4f} | "
          f"expected={EXPECTED_AGG_F1:.4f} | {status}", flush=True)
    if not match:
        print(f"PROVENANCE MISMATCH — not saving per-subject CSVs. "
              f"Computed={macro_f1:.4f}, expected={EXPECTED_AGG_F1:.4f} "
              f"(tolerance=±{TOLERANCE})", file=sys.stderr)
        sys.exit(1)

    # ── Save per-subject CSV ─────────────────────────────────────────────────
    os.makedirs(PER_SUBJECT_DIR, exist_ok=True)
    csv_fields = ["model", "condition", "subject_id", "macro_f1", "accuracy", "n_epochs"]

    csv_path = os.path.join(PER_SUBJECT_DIR,
                            f"yasa_{args.modality.lower()}_per_subject.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(subject_rows)
    print(f"Written: {csv_path}  ({len(subject_rows)} subjects)", flush=True)

    # ── For EEG_ONLY: also write EEG_ECG copy ───────────────────────────────
    if args.modality == "EEG_ONLY":
        eeg_ecg_rows = [{**r, "condition": "EEG_ECG"} for r in subject_rows]
        eeg_ecg_path = os.path.join(PER_SUBJECT_DIR, "yasa_eeg_ecg_per_subject.csv")
        with open(eeg_ecg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(eeg_ecg_rows)
        print(
            f"Written: {eeg_ecg_path}  ({len(eeg_ecg_rows)} subjects)\n"
            f"  NOTE: yasa_eeg_ecg_per_subject.csv is an intentional duplicate of\n"
            f"  yasa_eeg_only_per_subject.csv (condition column changed to EEG_ECG).\n"
            f"  YASA ignores ECG; predictions are identical for both conditions.",
            flush=True,
        )

    # ── Provenance report ────────────────────────────────────────────────────
    report_path = os.path.join(PER_SUBJECT_DIR, "yasa_provenance_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join([
            "YASA Per-Subject Evaluation — Provenance Report",
            "=" * 60,
            f"Run date: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"Modality run: {args.modality}",
            f"Split: {SPLIT_PATH}",
            f"Fold: {args.fold_key}",
            "",
            f"Subjects in split: {len(test_files)}",
            f"Subjects evaluated: {len(subject_rows)}",
            f"Subjects skipped: {n_skipped}  {skipped_sids}",
            "",
            "IMPORTANT — Subject-set mismatch vs. other models:",
            "  YASA has 49 subjects; BIOT/LaBraM/MOMENT/SensorLM all have 50.",
            "  mesa-sleep-0555 is absent from YASA because its HDF5 file is",
            "  missing on disk.  Any join on subject_id with other per-subject",
            "  CSVs will silently drop this subject for YASA.",
            "",
            f"Aggregate macro F1: {macro_f1:.4f} | expected: {EXPECTED_AGG_F1:.4f} | {status}",
            f"Aggregate accuracy: {acc:.4f}  n_epochs: {len(all_true_arr):,}",
            "",
            "EEG_ECG note:",
            "  yasa_eeg_ecg_per_subject.csv is a deliberate copy of",
            "  yasa_eeg_only_per_subject.csv with condition=EEG_ECG.",
            "  YASA uses only EEG1 regardless of modality label; ECG is ignored.",
        ]) + "\n")
    print(f"Provenance report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
