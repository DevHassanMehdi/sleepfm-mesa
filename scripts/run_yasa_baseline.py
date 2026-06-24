"""
YASA sleep staging baseline on MESA 10-fold CV.
Runs on raw EDF files, computes macro F1 against existing per-epoch labels.

Supports multiple modality configurations via --modality, for a 1-1
comparison with SleepFM's modality ablation. YASA's SleepStaging always
requires an EEG channel, so ECG_ONLY is not supported (see SKIP note below).
"""
import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
import mne
import yasa
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")

EDF_DIR = "/scratch/project_2019517/sleepfm-data/mesa/edf"
LABELS_DIR = "data/mesa/labels"
SPLIT_PATH = "data/mesa/dataset_split_10fold.json"

YASA_TO_INT = {"WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "ART": -1, "UNS": -1}
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# YASA's SleepStaging classifier is EEG-anchored (eeg_name is mandatory);
# it cannot run on ECG alone, hence no ECG_ONLY entry here.
MODALITY_KWARGS = {
    "EEG_EOG": dict(eeg_name="EEG1", eog_name="EOG-L", emg_name="EMG"),
    "EEG_ONLY": dict(eeg_name="EEG1", emg_name="EMG"),
}
MODALITY_CHANNELS_LABEL = {
    "EEG_EOG": "EEG1 + EOG-L + EMG",
    "EEG_ONLY": "EEG1 + EMG (no EOG)",
}

def get_subject_id(filepath):
    return os.path.basename(filepath).replace(".hdf5", "")

def load_labels(subject_id):
    label_path = os.path.join(LABELS_DIR, f"{subject_id}.csv")
    if not os.path.exists(label_path):
        return None
    return pd.read_csv(label_path)["StageNumber"].values.astype(int)

def run_yasa_on_subject(subject_id, modality_kwargs):
    edf_path = os.path.join(EDF_DIR, f"{subject_id}.edf")
    if not os.path.exists(edf_path):
        return None, f"EDF not found"
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        sls = yasa.SleepStaging(raw, **modality_kwargs)
        hypnogram = sls.predict()
        predicted = hypnogram.hypno.values
        pred_int = np.array([YASA_TO_INT[s] for s in predicted])
        return pred_int, None
    except Exception as e:
        return None, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="EEG_EOG",
                         choices=["EEG_EOG", "EEG_ONLY", "ECG_ONLY"])
    args = parser.parse_args()

    output_path = f"results/yasa_{args.modality}_results.txt"

    if args.modality == "ECG_ONLY":
        os.makedirs("results", exist_ok=True)
        msg = (
            "YASA BASELINE - ECG_ONLY\n"
            + "=" * 50 + "\n"
            "SKIPPED: yasa.SleepStaging requires an EEG channel (eeg_name is "
            "mandatory) and has no ECG-only mode. Not run.\n"
        )
        print(msg)
        with open(output_path, "w") as f:
            f.write(msg)
        return

    modality_kwargs = MODALITY_KWARGS[args.modality]

    with open(SPLIT_PATH) as f:
        splits = json.load(f)

    all_preds = []
    all_labels = []
    fold_f1s = []
    errors = []

    for fold_idx in range(10):
        fold_key = f"fold_{fold_idx}"
        test_files = splits[fold_key]["test"]
        fold_preds = []
        fold_labels = []

        print(f"\n=== Fold {fold_idx} ({len(test_files)} subjects) ===")

        for filepath in test_files:
            subject_id = get_subject_id(filepath)
            labels = load_labels(subject_id)
            if labels is None:
                print(f"  No labels: {subject_id}")
                continue

            preds, err = run_yasa_on_subject(subject_id, modality_kwargs)
            if preds is None:
                print(f"  Error {subject_id}: {err}")
                errors.append((subject_id, err))
                continue

            n = min(len(preds), len(labels))
            preds = preds[:n]
            labels = labels[:n]

            # exclude ART/UNS
            mask = preds >= 0
            fold_preds.append(preds[mask])
            fold_labels.append(labels[mask])
            subj_f1 = f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
            print(f"  {subject_id}: {subj_f1:.3f}", flush=True)

        if fold_preds:
            fp = np.concatenate(fold_preds)
            fl = np.concatenate(fold_labels)
            macro_f1 = f1_score(fl, fp, average="macro", zero_division=0)
            fold_f1s.append(macro_f1)
            all_preds.append(fp)
            all_labels.append(fl)
            print(f"  Fold {fold_idx} macro F1: {macro_f1:.4f}")

    all_preds_cat = np.concatenate(all_preds)
    all_labels_cat = np.concatenate(all_labels)
    overall_f1 = f1_score(all_labels_cat, all_preds_cat, average="macro", zero_division=0)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    report = classification_report(
        all_labels_cat, all_preds_cat,
        labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES,
        zero_division=0
    )

    lines = [
        f"YASA BASELINE ({args.modality}) — MESA 10-fold CV",
        "=" * 50,
        f"Channels: {MODALITY_CHANNELS_LABEL[args.modality]}",
        "",
        "Per-fold macro F1:",
    ]
    for i, f1 in enumerate(fold_f1s):
        lines.append(f"  Fold {i}: {f1:.4f}")
    lines += [
        f"\nMean fold macro F1: {mean_f1:.4f} +/- {std_f1:.4f}",
        f"Overall macro F1:   {overall_f1:.4f}",
        "",
        "Classification report (all folds combined):",
        report,
    ]
    if errors:
        lines.append(f"Errors ({len(errors)} subjects):")
        for sid, err in errors:
            lines.append(f"  {sid}: {err}")

    result_text = "\n".join(lines)
    print("\n" + result_text)
    with open(output_path, "w") as f:
        f.write(result_text)
    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
