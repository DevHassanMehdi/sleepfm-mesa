#!/usr/bin/env python3
"""
Run inference on the test split for BIOT, LaBraM, MOMENT, and SensorLM
and produce per-subject macro F1 / accuracy tables, with provenance checking
against ALL_RESULTS.md aggregate values.

Expected aggregate macro F1 (from ALL_RESULTS.md Section 3):
  BIOT   EEG_ONLY=0.7237  ECG_ONLY=0.3086  EEG_ECG=0.7023
  LaBraM EEG_ONLY=0.6835  ECG_ONLY=0.2803  EEG_ECG=0.6524
  MOMENT EEG_ONLY=0.5894  ECG_ONLY=0.3096  EEG_ECG=0.5953
  SensorLM EEG_ONLY=0.6264 ECG_ONLY=0.2821 EEG_ECG=0.6077

Outputs (one CSV per model × condition):
  results/per_subject/{model}_{condition}_per_subject.csv
  results/per_subject/baselines_provenance_report.txt
"""

import argparse
import csv
import os
import sys
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

OUT_DIR = REPO_ROOT / "results" / "per_subject"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE = 1e-3

EXPECTED_F1 = {
    "BIOT":     {"EEG_ONLY": 0.7237, "ECG_ONLY": 0.3086, "EEG_ECG": 0.7023},
    "LaBraM":   {"EEG_ONLY": 0.6835, "ECG_ONLY": 0.2803, "EEG_ECG": 0.6524},
    "MOMENT":   {"EEG_ONLY": 0.5894, "ECG_ONLY": 0.3096, "EEG_ECG": 0.5953},
    "SensorLM": {"EEG_ONLY": 0.6264, "ECG_ONLY": 0.2821, "EEG_ECG": 0.6077},
}

# Mapping from dataset modality key -> condition label in the CSV
MODALITY_TO_CONDITION = {
    "EEG_ONLY": "EEG_ONLY",
    "ECG_ONLY": "ECG_ONLY",
    "EEG_ECG":  "EEG_ECG",
}


# ─── helpers ────────────────────────────────────────────────────────────────

def epoch_preds_by_subject(dataset, all_preds_flat, all_targets_flat):
    """
    Group epoch-level predictions by subject using dataset.index.
    Returns list of (subject_id, t_arr, p_arr) sorted by subject_id.
    """
    assert len(dataset.index) == len(all_preds_flat) == len(all_targets_flat), \
        f"Length mismatch: index={len(dataset.index)} preds={len(all_preds_flat)}"

    buckets_t = defaultdict(list)
    buckets_p = defaultdict(list)
    for i, (hdf5_path, _, _) in enumerate(dataset.index):
        sid = os.path.basename(hdf5_path).replace(".hdf5", "")
        buckets_t[sid].append(int(all_targets_flat[i]))
        buckets_p[sid].append(int(all_preds_flat[i]))

    results = []
    for sid in sorted(buckets_t):
        t = np.array(buckets_t[sid], dtype=int)
        p = np.array(buckets_p[sid], dtype=int)
        results.append((sid, t, p))
    return results


def run_and_group(model_fn, dataset, batch_size, num_workers, device):
    """
    Runs inference, returns (all_preds_flat, all_targets_flat) aligned with dataset.index.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_preds, all_targets = [], []
    model = model_fn()
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  inference"):
            x, y = batch[0], batch[1]
            x = x.to(device)
            logits = model(x)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_targets.extend(y.numpy().tolist() if hasattr(y, "numpy") else list(y))
    return np.array(all_preds, dtype=int), np.array(all_targets, dtype=int)


# ─── BIOT ───────────────────────────────────────────────────────────────────

def evaluate_biot(modalities, fold_key, batch_size, num_workers, device, prov):
    sys.path.insert(0, "/scratch/project_2019517/BIOT")
    from biot_dataset import BIOTSleepDataset, MODALITY_CHANNELS
    from finetune_biot import build_model, CKPT_ROOT, SPLIT_PATH

    rows = []
    for mod in modalities:
        condition = MODALITY_TO_CONDITION[mod]
        ckpt_path = os.path.join(CKPT_ROOT, mod, fold_key, "best.pth")
        prov.append(f"\n=== BIOT / {mod} ===")
        prov.append(f"  Checkpoint: {ckpt_path}")
        prov.append(f"  mtime: {_mtime(ckpt_path)}")

        ds = BIOTSleepDataset(SPLIT_PATH, "test", mod, fold_key=fold_key)

        def model_fn():
            m = build_model()
            m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            return m

        preds_flat, targets_flat = run_and_group(model_fn, ds, batch_size, num_workers, device)
        agg_f1 = f1_score(targets_flat, preds_flat, average="macro", zero_division=0)
        agg_acc = accuracy_score(targets_flat, preds_flat)
        expected = EXPECTED_F1["BIOT"][condition]
        match = abs(agg_f1 - expected) <= TOLERANCE
        status = "MATCH" if match else "MISMATCH"
        prov.append(f"  Aggregate macro F1: {agg_f1:.4f} | expected: {expected:.4f} | {status}")
        prov.append(f"  Aggregate accuracy: {agg_acc:.4f}  n_epochs: {len(targets_flat):,}")
        if not match:
            print(f"PROVENANCE MISMATCH BIOT/{mod}: {agg_f1:.4f} != {expected:.4f}", file=sys.stderr)
            sys.exit(1)

        for sid, t, p in epoch_preds_by_subject(ds, preds_flat, targets_flat):
            rows.append({
                "model": "BIOT", "condition": condition, "subject_id": sid,
                "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
                "accuracy": round(float(accuracy_score(t, p)), 6),
                "n_epochs": len(t),
            })
    return rows


# ─── LaBraM ─────────────────────────────────────────────────────────────────

def evaluate_labram(modalities, fold_key, batch_size, num_workers, device, prov):
    sys.path.insert(0, "/scratch/project_2019517/LaBraM")
    from labram_dataset import LaBraMSleepDataset, MODALITY_CHANNELS, get_ch_names
    from finetune_labram import build_model, forward_logits, get_input_chans, CKPT_ROOT, SPLIT_PATH

    rows = []
    for mod in modalities:
        condition = MODALITY_TO_CONDITION[mod]
        ckpt_path = os.path.join(CKPT_ROOT, mod, fold_key, "best.pth")
        prov.append(f"\n=== LaBraM / {mod} ===")
        prov.append(f"  Checkpoint: {ckpt_path}")
        prov.append(f"  mtime: {_mtime(ckpt_path)}")

        ds = LaBraMSleepDataset(SPLIT_PATH, "test", mod, fold_key=fold_key)
        input_chans = torch.tensor(get_input_chans(get_ch_names(mod)), device=device)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        model = build_model(mod).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"  LaBraM/{mod}"):
                x = x.to(device)
                logits = forward_logits(model, x, input_chans)
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                all_targets.extend(y.numpy().tolist())

        preds_flat   = np.array(all_preds, dtype=int)
        targets_flat = np.array(all_targets, dtype=int)
        agg_f1 = f1_score(targets_flat, preds_flat, average="macro", zero_division=0)
        agg_acc = accuracy_score(targets_flat, preds_flat)
        expected = EXPECTED_F1["LaBraM"][condition]
        match = abs(agg_f1 - expected) <= TOLERANCE
        status = "MATCH" if match else "MISMATCH"
        prov.append(f"  Aggregate macro F1: {agg_f1:.4f} | expected: {expected:.4f} | {status}")
        prov.append(f"  Aggregate accuracy: {agg_acc:.4f}  n_epochs: {len(targets_flat):,}")
        if not match:
            print(f"PROVENANCE MISMATCH LaBraM/{mod}: {agg_f1:.4f} != {expected:.4f}", file=sys.stderr)
            sys.exit(1)

        for sid, t, p in epoch_preds_by_subject(ds, preds_flat, targets_flat):
            rows.append({
                "model": "LaBraM", "condition": condition, "subject_id": sid,
                "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
                "accuracy": round(float(accuracy_score(t, p)), 6),
                "n_epochs": len(t),
            })
    return rows


# ─── MOMENT ─────────────────────────────────────────────────────────────────

def evaluate_moment(modalities, fold_key, batch_size, num_workers, device, prov):
    from moment_dataset import MESADataset, MODALITY_CHANNELS
    from finetune_moment import build_model, forward_logits, CKPT_ROOT, SPLIT_PATH

    rows = []
    for mod in modalities:
        condition = MODALITY_TO_CONDITION[mod]
        ckpt_path = os.path.join(CKPT_ROOT, mod, fold_key, "best.pth")
        prov.append(f"\n=== MOMENT / {mod} ===")
        prov.append(f"  Checkpoint: {ckpt_path}")
        prov.append(f"  mtime: {_mtime(ckpt_path)}")

        ds = MESADataset(SPLIT_PATH, "test", mod, fold_key=fold_key)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        model = build_model(mod).to(device)
        model.head.linear.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"  MOMENT/{mod}"):
                x = x.to(device)
                logits = forward_logits(model, x)
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                all_targets.extend(y.numpy().tolist())

        preds_flat   = np.array(all_preds, dtype=int)
        targets_flat = np.array(all_targets, dtype=int)
        agg_f1 = f1_score(targets_flat, preds_flat, average="macro", zero_division=0)
        agg_acc = accuracy_score(targets_flat, preds_flat)
        expected = EXPECTED_F1["MOMENT"][condition]
        match = abs(agg_f1 - expected) <= TOLERANCE
        status = "MATCH" if match else "MISMATCH"
        prov.append(f"  Aggregate macro F1: {agg_f1:.4f} | expected: {expected:.4f} | {status}")
        prov.append(f"  Aggregate accuracy: {agg_acc:.4f}  n_epochs: {len(targets_flat):,}")
        if not match:
            print(f"PROVENANCE MISMATCH MOMENT/{mod}: {agg_f1:.4f} != {expected:.4f}", file=sys.stderr)
            sys.exit(1)

        for sid, t, p in epoch_preds_by_subject(ds, preds_flat, targets_flat):
            rows.append({
                "model": "MOMENT", "condition": condition, "subject_id": sid,
                "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
                "accuracy": round(float(accuracy_score(t, p)), 6),
                "n_epochs": len(t),
            })
    return rows


# ─── SensorLM ───────────────────────────────────────────────────────────────

def evaluate_sensorlm(modalities, fold_key, batch_size, num_workers, device, prov):
    from sensorlm_dataset import SensorLMSleepDataset, MODALITY_CHANNELS
    from sensorlm_model import SensorLMEncoder
    from finetune_sensorlm import CKPT_ROOT, SPLIT_PATH

    rows = []
    for mod in modalities:
        condition = MODALITY_TO_CONDITION[mod]
        ckpt_path = os.path.join(CKPT_ROOT, mod, fold_key, "best.pth")
        prov.append(f"\n=== SensorLM / {mod} ===")
        prov.append(f"  Checkpoint: {ckpt_path}")
        prov.append(f"  mtime: {_mtime(ckpt_path)}")

        ds = SensorLMSleepDataset(SPLIT_PATH, "test", mod, fold_key)
        n_channels = len(MODALITY_CHANNELS[mod])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        model = SensorLMEncoder(n_channels=n_channels).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"  SensorLM/{mod}"):
                x = x.to(device)
                logits = model(x)
                all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                all_targets.extend(y.numpy().tolist() if hasattr(y, "numpy") else list(y))

        preds_flat   = np.array(all_preds, dtype=int)
        targets_flat = np.array(all_targets, dtype=int)
        agg_f1 = f1_score(targets_flat, preds_flat, average="macro", zero_division=0)
        agg_acc = accuracy_score(targets_flat, preds_flat)
        expected = EXPECTED_F1["SensorLM"][condition]
        match = abs(agg_f1 - expected) <= TOLERANCE
        status = "MATCH" if match else "MISMATCH"
        prov.append(f"  Aggregate macro F1: {agg_f1:.4f} | expected: {expected:.4f} | {status}")
        prov.append(f"  Aggregate accuracy: {agg_acc:.4f}  n_epochs: {len(targets_flat):,}")
        if not match:
            print(f"PROVENANCE MISMATCH SensorLM/{mod}: {agg_f1:.4f} != {expected:.4f}", file=sys.stderr)
            sys.exit(1)

        for sid, t, p in epoch_preds_by_subject(ds, preds_flat, targets_flat):
            rows.append({
                "model": "SensorLM", "condition": condition, "subject_id": sid,
                "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
                "accuracy": round(float(accuracy_score(t, p)), 6),
                "n_epochs": len(t),
            })
    return rows


# ─── utils ──────────────────────────────────────────────────────────────────

def _mtime(path):
    try:
        t = os.path.getmtime(path)
        return datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")
    except OSError:
        return "NOT FOUND"


def write_csv(rows, model_tag, condition, out_dir):
    fname = out_dir / f"{model_tag.lower()}_{condition.lower()}_per_subject.csv"
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "condition", "subject_id",
                           "macro_f1", "accuracy", "n_epochs"])
        writer.writeheader()
        writer.writerows(rows)
    return fname


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=["BIOT", "LaBraM", "MOMENT", "SensorLM"],
                        default=["BIOT", "LaBraM", "MOMENT", "SensorLM"])
    parser.add_argument("--modalities", nargs="+",
                        choices=["EEG_ONLY", "ECG_ONLY", "EEG_ECG"],
                        default=["EEG_ONLY", "ECG_ONLY", "EEG_ECG"])
    parser.add_argument("--fold_key", default="fold_0")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    prov = [
        "Baseline Per-Subject Evaluation — Provenance Report",
        "=" * 60,
        f"Run date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Device: {device}",
    ]

    # Map condition labels used in EXPECTED_F1 to modality keys used in dataset scripts
    mod_map = {
        "EEG_ONLY": "EEG_ONLY",
        "ECG_ONLY": "ECG_ONLY",
        "EEG_ECG":  "EEG_ECG",
    }
    modalities = [mod_map[c] for c in args.modalities]

    def flush_model_csvs(model_tag, rows):
        """Write per-condition CSVs for a model immediately after inference completes."""
        prov.append(f"\nFiles written ({model_tag}):")
        for condition in args.modalities:
            subset = [r for r in rows if r["condition"] == condition]
            if not subset:
                continue
            out_csv = OUT_DIR / f"{model_tag.lower()}_{condition.lower()}_per_subject.csv"
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["model", "condition", "subject_id",
                                   "macro_f1", "accuracy", "n_epochs"])
                writer.writeheader()
                writer.writerows(subset)
            prov.append(f"  {out_csv}  ({len(subset)} subjects)")
            print(f"Written: {out_csv}", flush=True)

        report_path = OUT_DIR / "baselines_provenance_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(prov) + "\n")

    all_rows_by_model = {}

    if "BIOT" in args.models:
        rows = evaluate_biot(modalities, args.fold_key, 16, args.num_workers, device, prov)
        all_rows_by_model["BIOT"] = rows
        flush_model_csvs("BIOT", rows)

    if "LaBraM" in args.models:
        rows = evaluate_labram(modalities, args.fold_key, 16, args.num_workers, device, prov)
        all_rows_by_model["LaBraM"] = rows
        flush_model_csvs("LaBraM", rows)

    if "MOMENT" in args.models:
        rows = evaluate_moment(modalities, args.fold_key, 8, args.num_workers, device, prov)
        all_rows_by_model["MOMENT"] = rows
        flush_model_csvs("MOMENT", rows)

    if "SensorLM" in args.models:
        rows = evaluate_sensorlm(modalities, args.fold_key, 64, args.num_workers, device, prov)
        all_rows_by_model["SensorLM"] = rows
        flush_model_csvs("SensorLM", rows)

    # Combined CSV across all models
    all_rows = []
    for rows in all_rows_by_model.values():
        all_rows.extend(rows)
    combined_csv = OUT_DIR / "baselines_per_subject_combined.csv"
    all_rows.sort(key=lambda r: (r["model"], r["condition"], r["subject_id"]))
    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "condition", "subject_id",
                           "macro_f1", "accuracy", "n_epochs"])
        writer.writeheader()
        writer.writerows(all_rows)
    prov.append(f"  Combined: {combined_csv}  ({len(all_rows)} rows total)")
    print(f"Combined: {combined_csv}")

    report_path = OUT_DIR / "baselines_provenance_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(prov) + "\n")
    print(f"Provenance report: {report_path}")


if __name__ == "__main__":
    main()
