#!/usr/bin/env python3
"""
Extract per-subject macro F1 and accuracy for the SleepFM EEG_ONLY-encoder
fine-tuning results, with strict provenance checking against ALL_RESULTS.md.

Sources: all_outputs.pickle + all_targets.pickle + all_masks.pickle + all_paths.pickle
saved by evaluate_sleep_staging.py under
  /scratch/.../SleepEventLSTMClassifier_mesa_labels_{EEG_ONLY,EKG,EEG_ONLY_EKG}/fold_0/mesa/test/

Expected aggregate macro F1 (from ALL_RESULTS.md Section 2.2):
  EEG_ONLY    -> 0.6582
  EKG         -> 0.3353   (ECG_ONLY modality)
  EEG_ONLY_EKG -> 0.6529  (EEG+ECG modality)

Outputs:
  results/per_subject/sleepfm_eeg_only_per_subject.csv
  results/per_subject/sleepfm_ecg_only_per_subject.csv
  results/per_subject/sleepfm_eeg_ecg_per_subject.csv
  results/per_subject/sleepfm_provenance_report.txt
"""
import os
import sys
import pickle
import numpy as np
import csv
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_BASE = Path("/scratch/project_2019517/sleepfm-data/checkpoints")
OUT_DIR   = REPO_ROOT / "results" / "per_subject"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_F1 = {
    "EEG_ONLY":     0.6582,
    "EKG":          0.3353,
    "EEG_ONLY_EKG": 0.6529,
}

CONDITION_LABEL = {
    "EEG_ONLY":     "EEG_ONLY",
    "EKG":          "ECG_ONLY",
    "EEG_ONLY_EKG": "EEG_ECG",
}

TOLERANCE = 1e-3  # allow rounding in the 4th decimal place


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def per_subject_metrics(all_paths, all_outputs, all_targets, all_masks):
    """
    Each element of all_* corresponds to one DataLoader batch (batch_size=4
    in evaluate_sleep_staging.py). Outputs and targets are np arrays of shape
    [batch, max_seq_len, 5] and [batch, max_seq_len] respectively; masks are
    [batch, max_seq_len] with 0=valid, 1=padding.

    Returns list of dicts: {subject_id, macro_f1, accuracy, n_valid_windows}.
    """
    results = []
    for batch_idx, paths_batch in enumerate(all_paths):
        # paths_batch is a list of hdf5 paths for this batch
        n_in_batch = len(paths_batch)
        for j in range(n_in_batch):
            path = paths_batch[j]
            subj_outputs = all_outputs[batch_idx][j]  # [seq_len, 5]
            subj_targets = all_targets[batch_idx][j]  # [seq_len]
            subj_masks   = all_masks[batch_idx][j]    # [seq_len]

            valid = subj_masks == 0
            t = subj_targets[valid].astype(int)
            p = np.argmax(subj_outputs[valid], axis=1)

            if len(t) == 0:
                continue

            macro_f1 = f1_score(t, p, average="macro", zero_division=0)
            acc      = accuracy_score(t, p)
            subject_id = os.path.basename(path).replace(".hdf5", "")
            results.append({
                "subject_id": subject_id,
                "macro_f1": round(float(macro_f1), 6),
                "accuracy": round(float(acc), 6),
                "n_valid_windows": int(valid.sum()),
            })
    return results


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


def main():
    provenance_lines = [
        "SleepFM EEG_ONLY Encoder — Per-Subject Extraction Provenance Report",
        "=" * 70,
        "",
    ]

    csv_files = {}

    for key in ["EEG_ONLY", "EKG", "EEG_ONLY_EKG"]:
        label = CONDITION_LABEL[key]
        test_dir = CKPT_BASE / f"SleepEventLSTMClassifier_mesa_labels_{key}" / "fold_0" / "mesa" / "test"

        provenance_lines.append(f"=== Condition: {key} ({label}) ===")
        provenance_lines.append(f"  Directory: {test_dir}")

        # --- file existence check ---
        required = ["all_outputs.pickle", "all_targets.pickle",
                    "all_masks.pickle", "all_paths.pickle"]
        for fname in required:
            fpath = test_dir / fname
            if not fpath.exists():
                provenance_lines.append(f"  MISSING: {fname}")
                print(f"ERROR: {fpath} not found — aborting {key}", file=sys.stderr)
                continue
            stat = fpath.stat()
            import datetime
            mtime = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            provenance_lines.append(f"  {fname}: size={stat.st_size:,}B  mtime={mtime}")

        # --- load ---
        outputs_path = test_dir / "all_outputs.pickle"
        targets_path = test_dir / "all_targets.pickle"
        masks_path   = test_dir / "all_masks.pickle"
        paths_path   = test_dir / "all_paths.pickle"

        all_outputs = load_pickle(outputs_path)
        all_targets = load_pickle(targets_path)
        all_masks   = load_pickle(masks_path)
        all_paths   = load_pickle(paths_path)

        # --- provenance: n_subjects ---
        n_subjects = sum(len(b) for b in all_paths)
        provenance_lines.append(f"  Subjects in file: {n_subjects}")

        # --- aggregate F1 provenance check ---
        agg_f1, agg_acc, n_windows = compute_aggregate_f1(
            all_paths, all_outputs, all_targets, all_masks)
        expected = EXPECTED_F1[key]
        match = abs(agg_f1 - expected) <= TOLERANCE
        status = "MATCH" if match else "MISMATCH"
        provenance_lines.append(
            f"  Aggregate macro F1 (computed): {agg_f1:.4f} | expected: {expected:.4f} | {status}")
        provenance_lines.append(
            f"  Aggregate accuracy (computed): {agg_acc:.4f}  |  n_valid_windows: {n_windows:,}")
        provenance_lines.append("")

        if not match:
            print(
                f"PROVENANCE MISMATCH for {key}: computed F1={agg_f1:.4f} "
                f"!= expected {expected:.4f}. Stopping.",
                file=sys.stderr,
            )
            sys.exit(1)

        # --- per-subject extraction ---
        rows = per_subject_metrics(all_paths, all_outputs, all_targets, all_masks)
        rows.sort(key=lambda r: r["subject_id"])

        out_csv = OUT_DIR / f"sleepfm_{label.lower()}_per_subject.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["model", "condition", "subject_id",
                               "macro_f1", "accuracy", "n_valid_windows"])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "model": "SleepFM_EEG_ONLY_encoder",
                    "condition": label,
                    **row,
                })
        csv_files[key] = out_csv
        print(f"[{key}] Written {len(rows)} subjects to {out_csv}")

    provenance_lines.append("Files written:")
    for key, path in csv_files.items():
        provenance_lines.append(f"  {key}: {path}")

    report_path = OUT_DIR / "sleepfm_provenance_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(provenance_lines) + "\n")
    print(f"Provenance report: {report_path}")


if __name__ == "__main__":
    main()
