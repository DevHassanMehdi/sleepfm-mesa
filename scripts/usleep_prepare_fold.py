#!/usr/bin/env python3
"""
Prepare a U-Sleep "views" directory and dataset config for one fold of the
SleepFM 10-fold MESA split (data/mesa/dataset_split_10fold.json), so U-Sleep
trains/evaluates on exactly the same train/validation/test subjects as
SleepFM for a fair comparison.

For each subject in the fold's train/validation/test lists, creates a
subject directory under the fold's view tree containing symlinks to:
  - the raw EDF (psg.edf)
  - the converted .ids hypnogram (hypnogram.ids), see convert_labels_for_usleep.py

Also (over)writes <project_dir>/dataset_configurations/mesa.yaml so the
U-Sleep project's existing "mesa" dataset entry (left untouched in
hparams.yaml) points at this fold's view tree with the right channel names
and label mapping for MESA.

Usage:
    python scripts/usleep_prepare_fold.py --fold 0 \
        --project_dir /scratch/project_2019517/usleep_mesa/projects/fold_0
"""
import argparse
import json
import os
from pathlib import Path

SPLIT_JSON_TO_VIEW_DIR = {"train": "train", "validation": "val", "test": "test"}

EDF_DIR = Path("/scratch/project_2019517/sleepfm-data/mesa/edf")
HYPNOGRAM_DIR = Path("/scratch/project_2019517/usleep_mesa/hypnograms")
VIEWS_ROOT = Path("/scratch/project_2019517/usleep_mesa/views")

# Channel groups per modality. U-Sleep randomly samples one channel per group
# at train time and predicts on every combination (+ majority vote) at test
# time. The number of groups = number of input channels the model is built
# for (must match build.batch_shape's last dim, patched below).
MODALITY_CHANNEL_GROUPS = {
    "EEG_EOG": [["EEG1", "EEG2", "EEG3"], ["EOG-L", "EOG-R"]],
    "EEG_ONLY": [["EEG1", "EEG2", "EEG3"]],
    "ECG_ONLY": [["EKG"]],
    "EEG_ECG": [["EEG1", "EEG2", "EEG3"], ["EKG"]],
}


def format_channel_groups_yaml(groups) -> str:
    inner = ",\n  ".join(
        "[" + ", ".join(f"'{ch}'" for ch in group) + "]" for group in groups
    )
    return f"[\n  {inner}\n]"

DATASET_CONFIG_TEMPLATE = """\
train_data:
  data_dir: {views_dir}/train
  period_length: 30
  identifier: "TRAIN"
  psg_regex: .*[.]edf
  hyp_regex: .*[.]ids

val_data:
  data_dir: {views_dir}/val
  period_length: 30
  identifier: "VAL"
  psg_regex: .*[.]edf
  hyp_regex: .*[.]ids

test_data:
  data_dir: {views_dir}/test
  period_length: 30
  identifier: "TEST"
  psg_regex: .*[.]edf
  hyp_regex: .*[.]ids

set_sample_rate: 128

channel_sampling_groups: {channel_sampling_groups}

sleep_stage_annotations:
  W: 0
  N1: 1
  N2: 2
  N3: 3
  REM: 4
  UNKNOWN: 5

strip_func:
  strip_func: strip_to_match

quality_control_func:
  quality_control_func: "clip_noisy_values"
  min_max_times_global_iqr: 20

scaler: "RobustScaler"
batch_wise_scaling: false
"""


def subject_id_from_entry(entry: str) -> str:
    # entry looks like "mesa/mesa-sleep-0612.hdf5"
    return Path(entry).stem


def link_subject(subject_id: str, split_view_dir: Path):
    edf_path = EDF_DIR / f"{subject_id}.edf"
    ids_path = HYPNOGRAM_DIR / f"{subject_id}.ids"

    if not edf_path.exists():
        print(f"  SKIP {subject_id}: missing EDF at {edf_path}")
        return False
    if not ids_path.exists():
        print(f"  SKIP {subject_id}: missing hypnogram at {ids_path} (run convert_labels_for_usleep.py first)")
        return False

    subject_dir = split_view_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    edf_link = subject_dir / "psg.edf"
    ids_link = subject_dir / "hypnogram.ids"

    if edf_link.exists() or edf_link.is_symlink():
        edf_link.unlink()
    if ids_link.exists() or ids_link.is_symlink():
        ids_link.unlink()

    edf_link.symlink_to(edf_path)
    ids_link.symlink_to(ids_path)
    return True



def patch_hparams(project_dir: Path, n_channels: int):
    """Remove non-mesa datasets, cap n_epochs, and fix the model's expected
    input channel count (build.batch_shape's last dim) in hparams.yaml."""
    import re
    hparams_path = project_dir / "hyperparameters" / "hparams.yaml"
    if not hparams_path.exists():
        print(f"WARNING: hparams.yaml not found at {hparams_path}")
        return
    with open(hparams_path) as f:
        c = f.read()
    already_patched = "shhs:" not in c and "n_epochs: 500" in c
    if not already_patched:
        c = re.sub(
            r'datasets:.*?(?=\nbuild:)',
            'datasets:\n  mesa: dataset_configurations/mesa.yaml\n',
            c,
            flags=re.DOTALL
        )
        c = re.sub(r'n_epochs:\s*\d+', 'n_epochs: 500', c)
        c = re.sub(r',?\s*decay:\s*[0-9.]+', '', c)
        c = c.replace('ignore_out_of_bounds_classes: true', 'ignore_out_of_bounds_classes: false')
        c = re.sub(r'learning_rate:\s*[0-9e\.\-]+', 'learning_rate: 1.0e-04', c)
        c = re.sub(r'max_loaded_per_dataset:\s*\d+', 'max_loaded_per_dataset: 20', c)
        c = re.sub(r'num_access_before_reload:\s*\d+', 'num_access_before_reload: 64', c)

    # batch_shape: [batch, seq_len, samples_per_epoch, n_channels] - n_channels
    # must equal the number of channel_sampling_groups for this modality.
    c = re.sub(
        r'batch_shape:\s*\[(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,)\s*\d+\s*\]',
        lambda m: f'batch_shape: [{m.group(1)} {n_channels}]',
        c,
    )

    with open(hparams_path, "w") as f:
        f.write(c)
    print(f"hparams.yaml patched: mesa only, n_epochs=500, batch_shape n_channels={n_channels}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--modality", required=True, choices=sorted(MODALITY_CHANNEL_GROUPS),
                         help="Which channel set to configure U-Sleep for")
    parser.add_argument("--split_path", default="data/mesa/dataset_split_10fold.json")
    parser.add_argument("--project_dir", required=True,
                         help="Path to the U-Sleep project dir created by 'ut init'")
    args = parser.parse_args()

    full_split = json.load(open(args.split_path))
    fold_split = full_split[f"fold_{args.fold}"]

    views_dir = VIEWS_ROOT / f"fold_{args.fold}"

    for json_key, view_subdir in SPLIT_JSON_TO_VIEW_DIR.items():
        split_view_dir = views_dir / view_subdir
        split_view_dir.mkdir(parents=True, exist_ok=True)

        entries = fold_split[json_key]
        print(f"fold_{args.fold} / {json_key}: {len(entries)} subjects -> {split_view_dir}")

        n_ok = 0
        for entry in entries:
            subject_id = subject_id_from_entry(entry)
            if link_subject(subject_id, split_view_dir):
                n_ok += 1
        print(f"  linked {n_ok}/{len(entries)} subjects")

    channel_groups = MODALITY_CHANNEL_GROUPS[args.modality]

    project_dir = Path(args.project_dir)
    dataset_config_dir = project_dir / "hyperparameters" / "dataset_configurations"
    dataset_config_dir.mkdir(parents=True, exist_ok=True)
    dataset_config_path = dataset_config_dir / "mesa.yaml"
    dataset_config_path.write_text(
        DATASET_CONFIG_TEMPLATE.format(
            views_dir=views_dir,
            channel_sampling_groups=format_channel_groups_yaml(channel_groups),
        )
    )
    print(f"Wrote dataset config ({args.modality}, {len(channel_groups)} channel(s)): {dataset_config_path}")
    patch_hparams(project_dir, n_channels=len(channel_groups))


if __name__ == "__main__":
    main()
