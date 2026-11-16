"""Per-subject 5x5 confusion-matrix counts for the new primary statistic
(results/07 item 1): epoch-pooled macro F1, with a cluster bootstrap that
resamples subjects and re-pools COUNTS (not per-subject macro F1 values)
in each resample. Mirrors build_per_subject.py's raw-prediction iteration
exactly (same source files, same subject/fold attribution) but stores the
full confusion matrix per (model, condition, fold, subject_id) instead of
derived scalar metrics, so pooling-by-summation is exact for any subset of
subjects, not just the full cohort or a single fold.
"""
import glob
import os
import sys

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "/users/hamehd/projects/sleepfm-mesa/sleepfm")
from utils import load_data

REPO_ROOT = "/users/hamehd/projects/sleepfm-mesa"
CKPT = f"{REPO_ROOT}/data/checkpoints/full_cohort"
BIOT_PRED_DIR = os.path.expanduser("~/analysis/eeg_channels/biot_predictions")
LABELS = [0, 1, 2, 3, 4]

SLEEPFM_RUNS = {
    ("SleepFM From-Scratch", "EEG1 only"): f"{CKPT}/sleepfm__EEG1_ONLY__fromscratch__fold5_v1__2026-10-27_1521",
    ("SleepFM Next-Token", "EEG1 only"): f"{CKPT}/sleepfm__EEG1_ONLY__nexttoken__fold5_v1__2026-10-27_1522",
    ("SleepFM From-Scratch", "EEG2 only"): f"{CKPT}/sleepfm__EEG2_ONLY__fromscratch__fold5_v1__2026-10-27_153632",
    ("SleepFM Next-Token", "EEG2 only"): f"{CKPT}/sleepfm__EEG2_ONLY__nexttoken__fold5_v1__2026-10-27_153632_y",
    ("SleepFM From-Scratch", "EEG3 only"): f"{CKPT}/sleepfm__EEG3_ONLY__fromscratch__fold5_v1__2026-10-27_153632_x",
    ("SleepFM Next-Token", "EEG3 only"): f"{CKPT}/sleepfm__EEG3_ONLY__nexttoken__fold5_v1__2026-10-27_153632_z",
    ("SleepFM From-Scratch", "3-channel, patience 10"): f"{CKPT}/sleepfm__EEG_ONLY__fromscratch_p10ref__fold5_v1__2026-10-28_154103",
    ("SleepFM Next-Token", "3-channel, patience 10"): f"{CKPT}/sleepfm__EEG_ONLY__nexttoken_p10ref__fold5_v1__2026-10-28_154103_nt",
    ("SleepFM From-Scratch", "3-channel benchmark"): f"{CKPT}/sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-09-19_1657",
    ("SleepFM Next-Token", "3-channel benchmark"): f"{CKPT}/sleepfm__EEG_ONLY__nexttoken__fold5_v1__2026-09-17_2326",
}

BIOT_RUNS = {
    ("BIOT", "EEG1 only"): "biot__EEG1_ONLY__finetuned__fold5_v1",
    ("BIOT", "EEG2 only"): "biot__EEG2_ONLY__finetuned__fold5_v1",
    ("BIOT", "EEG3 only"): "biot__EEG3_ONLY__finetuned__fold5_v1",
    ("BIOT", "3-channel benchmark"): "biot__EEG_ONLY__finetuned__fold5_v1",
}


def process_sleepfm_run(model, condition, base_dir):
    subject_ids, folds, cms = [], [], []
    for fold in range(5):
        fold_dir = os.path.join(base_dir, f"fold_{fold}")
        outputs = load_data(os.path.join(fold_dir, "test_all_outputs.pickle"))
        targets = load_data(os.path.join(fold_dir, "test_all_targets.pickle"))
        masks = load_data(os.path.join(fold_dir, "test_all_masks.pickle"))
        paths = load_data(os.path.join(fold_dir, "test_all_paths.pickle"))
        for o_batch, t_batch, m_batch, p_batch in zip(outputs, targets, masks, paths):
            o_batch, t_batch, m_batch = np.array(o_batch), np.array(t_batch), np.array(m_batch)
            for i in range(o_batch.shape[0]):
                valid = m_batch[i] == 0
                if valid.sum() == 0:
                    continue
                preds_s = np.argmax(o_batch[i], axis=-1)[valid]
                targets_s = t_batch[i][valid].astype(int)
                subject_id = os.path.basename(str(p_batch[i])).replace(".hdf5", "")
                cm = confusion_matrix(targets_s, preds_s, labels=LABELS)
                subject_ids.append(subject_id)
                folds.append(fold)
                cms.append(cm)
    return subject_ids, folds, cms


def process_biot_run(model, condition, run_prefix):
    subject_ids, folds, cms = [], [], []
    pattern = os.path.join(BIOT_PRED_DIR, f"{run_prefix}__*", "fold_*", "predictions.npz")
    for npz_path in sorted(glob.glob(pattern)):
        fold = int(os.path.basename(os.path.dirname(npz_path)).replace("fold_", ""))
        d = np.load(npz_path, allow_pickle=True)
        labels, preds, subj_idx = d["labels"], d["predictions"], d["subject_index"]
        for subject_id in sorted(set(subj_idx.tolist())):
            sel = subj_idx == subject_id
            cm = confusion_matrix(labels[sel].astype(int), preds[sel].astype(int), labels=LABELS)
            subject_ids.append(subject_id)
            folds.append(fold)
            cms.append(cm)
    return subject_ids, folds, cms


def sanitize(model, condition):
    return (model + "__" + condition).replace(" ", "_").replace(",", "").replace("-", "_")


if __name__ == "__main__":
    out = {}
    for (model, condition), base_dir in SLEEPFM_RUNS.items():
        print(f"Processing {model} / {condition} ...", flush=True)
        subject_ids, folds, cms = process_sleepfm_run(model, condition, base_dir)
        key = sanitize(model, condition)
        out[f"subjects__{key}"] = np.array(subject_ids)
        out[f"fold__{key}"] = np.array(folds, dtype=np.int32)
        out[f"cm__{key}"] = np.array(cms, dtype=np.int64)
        print(f"  {len(subject_ids)} subject rows, cm shape {out[f'cm__{key}'].shape}")

    for (model, condition), prefix in BIOT_RUNS.items():
        print(f"Processing {model} / {condition} (from saved predictions) ...", flush=True)
        subject_ids, folds, cms = process_biot_run(model, condition, prefix)
        key = sanitize(model, condition)
        out[f"subjects__{key}"] = np.array(subject_ids)
        out[f"fold__{key}"] = np.array(folds, dtype=np.int32)
        out[f"cm__{key}"] = np.array(cms, dtype=np.int64)
        print(f"  {len(subject_ids)} subject rows, cm shape {out[f'cm__{key}'].shape}")

    out_dir = os.path.expanduser("~/analysis/eeg_channels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "per_subject_confusion.npz")
    np.savez_compressed(out_path, **out)
    print(f"\nWrote {out_path}")

    # Sanity check: pooling ALL subjects' cm for one (model,condition) and
    # taking macro F1 from the pooled matrix should reproduce the published
    # 3-channel-benchmark macro F1 (within rounding) -- same cross-check
    # Part C already did for the per-fold epoch-flattened statistic.
    def macro_f1_from_cm(cm):
        f1s = []
        for c in range(5):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = 2 * tp + fp + fn
            f1s.append(2 * tp / denom if denom > 0 else 0.0)
        return float(np.mean(f1s))

    print("\nRun-level (whole-cohort) epoch-pooled macro F1, sanity check:")
    for (model, condition) in list(SLEEPFM_RUNS.keys()) + list(BIOT_RUNS.keys()):
        key = sanitize(model, condition)
        pooled = out[f"cm__{key}"].sum(axis=0)
        print(f"  {model} / {condition}: {macro_f1_from_cm(pooled):.4f}  (n_subjects={out[f'cm__{key}'].shape[0]})")
