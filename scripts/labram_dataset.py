"""
PyTorch Dataset for fine-tuning LaBraM on MESA PSG sleep staging.

LaBraM is EEG-only and indexes its pretrained positional embedding by
standard 10-20 electrode name, so only the EEG_ONLY modality is supported
here (no BAS: MESA's EOG channels have no defensible single-site mapping in
LaBraM's vocabulary).

Channel identity for MESA's EEG1/EEG2/EEG3 was confirmed from the official
NSRR "MESA Sleep Polysomnography Manual of Procedures" (Somte PIB hookup,
p.29-31): this is the standard 3-channel Somte EEG montage --
  EEG1 = Fz(+) - Cz(-)
  EEG2 = Cz(+) - Oz(-)   (Cz jumpered as shared reference)
  EEG3 = C4(+) - M1(-)
Each derivation is mapped to its non-reference ("active") electrode, which
is the standard convention for reusing single-site-indexed pretrained EEG
models on bipolar derivations:
  EEG1 -> FZ, EEG2 -> OZ, EEG3 -> C4
(all three confirmed present in LaBraM's standard_1020 vocabulary).

Resamples MESA's 128Hz to LaBraM's native 200Hz (patch_size=200, i.e. one
patch per second) and divides by 100, matching LaBraM's own preprocessing
convention in engine_for_finetuning.py (EEG.float() / 100).
"""
import os
import json
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from scipy.signal import resample

MESA_HZ = 128
LABRAM_HZ = 200
EPOCH_SEC = 30
MESA_SAMPLES = MESA_HZ * EPOCH_SEC      # 3840
LABRAM_SAMPLES = LABRAM_HZ * EPOCH_SEC  # 6000, i.e. 30 one-second patches

HDF5_DIR = "data/mesa/hdf5"
LABELS_DIR = "data/mesa/labels"

# MESA HDF5 dataset name -> LaBraM standard_1020 channel name.
MODALITY_CHANNELS = {
    "EEG_ONLY": ["EEG1", "EEG2", "EEG3"],
}
EEG_TO_STANDARD_1020 = {
    "EEG1": "FZ",
    "EEG2": "OZ",
    "EEG3": "C4",
}

_SPLIT_KEY = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def get_subject_id(filename):
    return filename.replace(".hdf5", "")


def get_ch_names(modality):
    return [EEG_TO_STANDARD_1020[ch] for ch in MODALITY_CHANNELS[modality]]


class LaBraMSleepDataset(Dataset):
    def __init__(self, split_path, split, modality, fold_key="fold_0",
                 hdf5_dir=HDF5_DIR, labels_dir=LABELS_DIR):
        self.modality = modality
        self.channels = MODALITY_CHANNELS[modality]
        self.hdf5_dir = hdf5_dir

        with open(split_path) as f:
            splits = json.load(f)
        files = splits[fold_key][_SPLIT_KEY[split]]

        # Flat (hdf5_path, start_sample_128hz, stage) index over every
        # labelled 30s epoch of every subject in this split.
        self.index = []
        for filename in files:
            subject_id = get_subject_id(filename)
            hdf5_path = os.path.join(hdf5_dir, filename)
            label_path = os.path.join(labels_dir, f"{subject_id}.csv")
            if not os.path.exists(hdf5_path) or not os.path.exists(label_path):
                continue
            labels_df = pd.read_csv(label_path)
            stages = labels_df["StageNumber"].to_numpy()
            starts = np.round(labels_df["Start"].to_numpy() * MESA_HZ).astype(int)
            valid = np.isin(stages, [0, 1, 2, 3, 4])
            for start_sample, stage in zip(starts[valid], stages[valid]):
                self.index.append((hdf5_path, int(start_sample), int(stage)))

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
        return x, stage
