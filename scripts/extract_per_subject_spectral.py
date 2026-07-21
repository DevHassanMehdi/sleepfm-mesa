#!/usr/bin/env python3
"""
Extract per-subject macro F1 and accuracy for the SleepFM SPECTRAL encoder
fine-tuning results, with strict provenance checking against Spectral_ALL_results.txt.

Sources: all_outputs.pickle + all_targets.pickle + all_masks.pickle + all_paths.pickle
saved by evaluate_sleep_staging.py under
  /scratch/.../SleepEventLSTMClassifier_mesa_labels_{EEG_ONLY,EKG,EEG_ONLY_EKG}/fold_0/mesa/test/

Expected aggregate macro F1 (from results/Spectral_ALL_results.txt):
  EEG_ONLY     -> 0.6971
  EKG          -> 0.2603   (ECG_ONLY modality)
  EEG_ONLY_EKG -> 0.6875   (EEG+ECG modality)

Outputs:
  results/per_subject/sleepfm_spectral_eeg_only_per_subject.csv
  results/per_subject/sleepfm_spectral_ecg_only_per_subject.csv
  results/per_subject/sleepfm_spectral_eeg_ecg_per_subject.csv
  results/per_subject/sleepfm_spectral_provenance_report.txt

All three conditions must pass provenance before any CSV is written.
"""
import csv
import datetime
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_BASE = Path("/scratch/project_2019517/sleepfm-data/checkpoints")
OUT_DIR   = REPO_ROOT / "results" / "per_subject"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_F1 = {
    "EEG_ONLY":     0.6971,
    "EKG":          0.2603,
    "EEG_ONLY_EKG": 0.6875,
}

CONDITION_LABEL = {
    "EEG_ONLY":     "EEG_ONLY",
    "EKG":          "ECG_ONLY",
    "EEG_ONLY_EKG": "EEG_ECG",
}

TOLERANCE = 1e-3


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_aggregate_f1(all_paths, all_outputs, all_targets, all_masks):
    all_t, all_p = [], []
    for batch_idx, paths_batch in enumerate(all_paths):
        for j in range(len(paths_batch)):
            valid = all_masks[batch_idx][j] == 0
            t = all_targets[batch_idx][j][valid].astype(int)
            p = np.argmax(all_outputs[batch_idx][j][valid], axis=1)
            all_t.extend(t.tolist())
            all_p.extend(p.tolist())
    all_t = np.array(all_t, dtype=int)
    all_p = np.array(all_p, dtype=int)
    return (
        f1_score(all_t, all_p, average="macro", zero_division=0),
        accuracy_score(all_t, all_p),
        len(all_t),
    )


def per_subject_metrics(all_paths, all_outputs, all_targets, all_masks):
    results = []
    for batch_idx, paths_batch in enumerate(all_paths):
        for j in range(len(paths_batch)):
            path = paths_batch[j]
            subj_outputs = all_outputs[batch_idx][j]
            subj_targets = all_targets[batch_idx][j]
            subj_masks   = all_masks[batch_idx][j]

            valid = subj_masks == 0
            t = subj_targets[valid].astype(int)
            p = np.argmax(subj_outputs[valid], axis=1)

            if len(t) == 0:
                continue

            results.append({
                "subject_id":       os.path.basename(path).replace(".hdf5", ""),
                "macro_f1":         round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
                "accuracy":         round(float(accuracy_score(t, p)), 6),
                "n_valid_windows":  int(valid.sum()),
            })
    return results


def main():
    prov = [
        "SleepFM SPECTRAL Encoder — Per-Subject Extraction Provenance Report",
        "=" * 70,
        f"Run date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    # ── Pass 1: load all three conditions and verify provenance ───────────────
    # Do NOT write any CSV until ALL three conditions pass.
    loaded = {}
    all_matched = True

    for key in ["EEG_ONLY", "EKG", "EEG_ONLY_EKG"]:
        label    = CONDITION_LABEL[key]
        test_dir = CKPT_BASE / f"SleepEventLSTMClassifier_mesa_labels_{key}" / "fold_0" / "mesa" / "test"

        prov.append(f"=== Condition: {key} ({label}) ===")
        prov.append(f"  Directory: {test_dir}")

        required = ["all_outputs.pickle", "all_targets.pickle",
                    "all_masks.pickle", "all_paths.pickle"]
        for fname in required:
            fpath = test_dir / fname
            if not fpath.exists():
                print(f"ERROR: {fpath} not found", file=sys.stderr)
                sys.exit(1)
            mtime = datetime.datetime.fromtimestamp(fpath.stat().st_mtime).isoformat(timespec="seconds")
            prov.append(f"  {fname}: size={fpath.stat().st_size:,}B  mtime={mtime}")

        all_outputs = load_pickle(test_dir / "all_outputs.pickle")
        all_targets = load_pickle(test_dir / "all_targets.pickle")
        all_masks   = load_pickle(test_dir / "all_masks.pickle")
        all_paths   = load_pickle(test_dir / "all_paths.pickle")

        n_subjects = sum(len(b) for b in all_paths)
        prov.append(f"  Subjects in file: {n_subjects}")

        agg_f1, agg_acc, n_windows = compute_aggregate_f1(
            all_paths, all_outputs, all_targets, all_masks)
        expected = EXPECTED_F1[key]
        match    = abs(agg_f1 - expected) <= TOLERANCE
        status   = "MATCH" if match else "MISMATCH"
        prov.append(f"  Aggregate macro F1: {agg_f1:.4f} | expected: {expected:.4f} | {status}")
        prov.append(f"  Aggregate accuracy: {agg_acc:.4f}  n_valid_windows: {n_windows:,}")
        prov.append("")

        print(f"[{key}] macro F1={agg_f1:.4f} | expected={expected:.4f} | {status}", flush=True)

        if not match:
            all_matched = False
            print(f"PROVENANCE MISMATCH for {key}: computed={agg_f1:.4f} != expected={expected:.4f}",
                  file=sys.stderr)

        loaded[key] = (all_paths, all_outputs, all_targets, all_masks)

    if not all_matched:
        prov.append("RESULT: MISMATCH detected — no CSVs written.")
        report_path = OUT_DIR / "sleepfm_spectral_provenance_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(prov) + "\n")
        print(f"Provenance report (mismatch): {report_path}", file=sys.stderr)
        sys.exit(1)

    # ── Pass 2: all conditions matched — write CSVs ───────────────────────────
    csv_fields = ["model", "condition", "subject_id", "macro_f1", "accuracy", "n_valid_windows"]
    prov.append("RESULT: All conditions MATCH — writing CSVs.")
    prov.append("")
    prov.append("Files written:")

    for key, (all_paths, all_outputs, all_targets, all_masks) in loaded.items():
        label = CONDITION_LABEL[key]
        rows  = per_subject_metrics(all_paths, all_outputs, all_targets, all_masks)
        rows.sort(key=lambda r: r["subject_id"])

        out_csv = OUT_DIR / f"sleepfm_spectral_{label.lower()}_per_subject.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({"model": "SleepFM_Spectral", "condition": label, **row})

        prov.append(f"  {out_csv}  ({len(rows)} subjects)")
        print(f"Written: {out_csv}  ({len(rows)} subjects)", flush=True)

    report_path = OUT_DIR / "sleepfm_spectral_provenance_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(prov) + "\n")
    print(f"Provenance report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
