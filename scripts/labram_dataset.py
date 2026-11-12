"""
PyTorch Dataset for fine-tuning LaBraM on MESA PSG sleep staging.

LaBraM indexes its pretrained positional embedding by standard 10-20
electrode name via get_input_chans(). EEG channels are mapped based on
the confirmed NSRR MESA PSG montage; non-EEG channels are assigned to
unused peripheral positions in LaBraM's standard_1020 vocabulary as a
forced experimental mapping (see CHANNEL_TO_STANDARD_1020 below).

== EEG mapping (physiologically grounded) ==
Confirmed from NSRR "MESA Sleep Polysomnography Manual of Procedures"
(Somte PIB hookup, pp.29-31) — standard 3-channel Somte EEG montage:
  EEG1 = Fz(+) - Cz(-)   ->  FZ   (active electrode)
  EEG2 = Cz(+) - Oz(-)   ->  OZ   (active electrode)
  EEG3 = C4(+) - M1(-)   ->  C4   (active electrode)

== Non-EEG channel mapping (forced / research-only) ==
EOG, ECG, respiratory, and EMG channels have no physiologically meaningful
10-20 site. They are assigned to unused peripheral positions (T9, T10, …)
purely to satisfy LaBraM's positional embedding index. Results for
modalities that include non-EEG channels are intentionally experimental
and show what happens when an EEG-specific foundation model is forced to
process non-EEG signals.

  EOG-L  ->  FP1    EOG-R  ->  FP2
  EKG    ->  T9
  Abdo   ->  T10    HR     ->  TP9    Snore  ->  TP10
  SpO2   ->  P9     Therm  ->  P10    Thor   ->  PO9
  EMG    ->  PO10   Leg    ->  O9     Pleth  ->  O10

All 15 positions are valid entries in LaBraM's standard_1020 list and
none collide with the EEG slots (FZ, OZ, C4).

Preprocessing: resample MESA 128 Hz -> LaBraM 200 Hz (patch_size=200,
one patch per second). No division by 100 here; finetune_labram.py does
that in forward_logits() to match LaBraM's engine_for_finetuning.py.
"""
import os
import json
import importlib.util
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import resample

# Callers (evaluate_labram.py/finetune_labram.py) add scripts/, sleepfm/,
# and /scratch/project_2019517/LaBraM to sys.path, and LaBraM has its own
# unrelated utils.py -- whichever of the two "utils" modules happens to
# already be cached in sys.modules (by another file's import order)
# silently wins a bare `import utils`/`from utils import ...` here. Load
# sleepfm's copy from its exact file path instead, immune to sys.path/
# sys.modules order.
_SLEEPFM_UTILS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sleepfm", "utils.py")
_spec = importlib.util.spec_from_file_location("sleepfm_utils_for_labram_dataset", _SLEEPFM_UTILS_PATH)
_sleepfm_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sleepfm_utils)
check_subject_count = _sleepfm_utils.check_subject_count

MESA_HZ = 128
LABRAM_HZ = 200
EPOCH_SEC = 30
MESA_SAMPLES = MESA_HZ * EPOCH_SEC      # 3840
LABRAM_SAMPLES = LABRAM_HZ * EPOCH_SEC  # 6000, i.e. 30 one-second patches

HDF5_DIR = "data/mesa/hdf5"
LABELS_DIR = "data/mesa/labels"

MODALITY_CHANNELS = {
    "EEG_ONLY":         ["EEG1", "EEG2", "EEG3"],
    "ECG_ONLY":         ["EKG"],
    "EEG_ECG":          ["EEG1", "EEG2", "EEG3", "EKG"],
    "BAS":              ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R"],
    "BAS_EKG":          ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R", "EKG"],
    "BAS_EKG_RESP":     ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R", "EKG",
                          "Abdo", "HR", "Snore", "SpO2", "Therm", "Thor"],
    "BAS_EKG_RESP_EMG": ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R", "EKG",
                          "Abdo", "HR", "Snore", "SpO2", "Therm", "Thor",
                          "EMG", "Leg", "Pleth"],
}

# MESA HDF5 channel name -> LaBraM standard_1020 electrode name.
# EEG entries are physiologically grounded (see module docstring).
# Non-EEG entries are forced placeholder mappings for research experiments.
CHANNEL_TO_STANDARD_1020 = {
    # EEG — confirmed from NSRR MESA PSG Manual pp.29-31
    "EEG1":  "FZ",
    "EEG2":  "OZ",
    "EEG3":  "C4",
    # EOG — assigned to frontal periphery (closest anatomical neighbours)
    "EOG-L": "FP1",
    "EOG-R": "FP2",
    # ECG — assigned to left lateral periphery
    "EKG":   "T9",
    # Respiratory — assigned to posterior peripheral ring
    "Abdo":  "T10",
    "HR":    "TP9",
    "Snore": "TP10",
    "SpO2":  "P9",
    "Therm": "P10",
    "Thor":  "PO9",
    # EMG / movement / pulse — assigned to occipital periphery
    "EMG":   "PO10",
    "Leg":   "O9",
    "Pleth": "O10",
}

_SPLIT_KEY = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def get_subject_id(filename):
    return filename.replace(".hdf5", "")


def get_ch_names(modality):
    return [CHANNEL_TO_STANDARD_1020[ch] for ch in MODALITY_CHANNELS[modality]]


class LaBraMSleepDataset(Dataset):
    def __init__(self, split_path, split, modality, fold_key="fold_0",
                 hdf5_dir=HDF5_DIR, labels_dir=LABELS_DIR, skip_unscored=False,
                 allow_missing_subjects=False):
        self.modality = modality
        self.channels = MODALITY_CHANNELS[modality]
        self.hdf5_dir = hdf5_dir

        with open(split_path) as f:
            splits = json.load(f)
        files = splits[fold_key][_SPLIT_KEY[split]]

        # Flat (hdf5_path, start_sample_128hz, stage) index over every
        # labelled 30s epoch of every subject in this split.
        self.index = []
        self.skipped_subjects = []
        self.missing_files = []
        for filename in files:
            subject_id = get_subject_id(filename)
            hdf5_path = os.path.join(hdf5_dir, filename)
            label_path = os.path.join(labels_dir, f"{subject_id}.csv")
            if not os.path.exists(hdf5_path) or not os.path.exists(label_path):
                self.missing_files.append((subject_id, hdf5_path, label_path))
                continue
            with h5py.File(hdf5_path, "r") as hf:
                missing = [ch for ch in self.channels if ch not in hf]
            if missing:
                print(f"WARNING: skipping {subject_id} ({split}, modality={modality}): "
                      f"missing channel(s) {missing}")
                self.skipped_subjects.append((subject_id, missing))
                continue
            labels_df = pd.read_csv(label_path)
            stages = labels_df["StageNumber"].to_numpy()
            starts = np.round(labels_df["Start"].to_numpy() * MESA_HZ).astype(int)
            # np.isin(..., [0,1,2,3,4]) already excludes -1 (unscored)
            # unconditionally, same as any other out-of-range value --
            # skip_unscored makes that exclusion explicit/toggleable
            # rather than an implicit side effect of the broader filter,
            # for datasets (e.g. Moody/PhysioNet 2026) that use -1 for
            # unscored epochs. A no-op either way for MESA, which never
            # has -1 in StageNumber.
            valid = np.isin(stages, [0, 1, 2, 3, 4])
            if skip_unscored:
                n_unscored = int((stages == -1).sum())
                if n_unscored:
                    print(f"skip_unscored: excluding {n_unscored} unscored "
                          f"(-1) epochs for {subject_id}")
                valid = valid & (stages != -1)
            for start_sample, stage in zip(starts[valid], stages[valid]):
                self.index.append((hdf5_path, int(start_sample), int(stage)))

        n_loaded = len(files) - len(self.missing_files) - len(self.skipped_subjects)
        check_subject_count(f"LaBraM {modality} {split}", len(files), n_loaded,
                             hdf5_dir, labels_dir, allow_missing=allow_missing_subjects)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        hdf5_path, start_sample, stage = self.index[idx]
        end_sample = start_sample + MESA_SAMPLES
        x = np.zeros((len(self.channels), MESA_SAMPLES), dtype=np.float32)
        with h5py.File(hdf5_path, "r") as hf:
            for i, ch in enumerate(self.channels):
                signal = hf[ch][start_sample:end_sample]
                x[i, :len(signal)] = signal
        x = resample(x, LABRAM_SAMPLES, axis=-1)
        x = torch.from_numpy(x.astype(np.float32))
        return x, stage, hdf5_path
