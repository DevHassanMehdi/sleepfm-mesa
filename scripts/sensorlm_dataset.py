"""MESA HDF5 dataset for SensorLM fine-tuning — same pipeline as BIOT/MOMENT."""
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MESA_HZ    = 128
EPOCH_SEC  = 30
EPOCH_SAMP = MESA_HZ * EPOCH_SEC   # 3840

HDF5_DIR   = "data/mesa/hdf5"
LABELS_DIR = "data/mesa/labels"

MODALITY_CHANNELS = {
    "EEG_ONLY": ["EEG1", "EEG2", "EEG3"],
    "ECG_ONLY": ["EKG"],
    "EEG_ECG":  ["EEG1", "EEG2", "EEG3", "EKG"],
}

_SPLIT_KEY = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


class SensorLMSleepDataset(Dataset):
    def __init__(self, split_path, split, modality, fold_key="fold_0",
                 hdf5_dir=HDF5_DIR, labels_dir=LABELS_DIR):
        self.modality = modality
        self.channels = MODALITY_CHANNELS[modality]
        self.hdf5_dir = hdf5_dir

        with open(split_path) as f:
            files = json.load(f)[fold_key][_SPLIT_KEY[split]]

        self.index = []
        self.skipped_subjects = []
        for fname in files:
            sid        = fname.replace(".hdf5", "")
            hdf5_path  = os.path.join(hdf5_dir, fname)
            label_path = os.path.join(labels_dir, f"{sid}.csv")
            if not os.path.exists(hdf5_path) or not os.path.exists(label_path):
                continue
            with h5py.File(hdf5_path, "r") as hf:
                missing = [ch for ch in self.channels if ch not in hf]
            if missing:
                self.skipped_subjects.append((sid, missing))
                continue
            df     = pd.read_csv(label_path)
            valid  = np.isin(df["StageNumber"].to_numpy(), [0, 1, 2, 3, 4])
            starts = np.round(df["Start"].to_numpy() * MESA_HZ).astype(int)
            stages = df["StageNumber"].to_numpy()
            for s, st in zip(starts[valid], stages[valid]):
                self.index.append((hdf5_path, int(s), int(st)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        hdf5_path, start, stage = self.index[idx]
        end = start + EPOCH_SAMP
        x   = np.zeros((len(self.channels), EPOCH_SAMP), dtype=np.float32)
        with h5py.File(hdf5_path, "r") as hf:
            for i, ch in enumerate(self.channels):
                sig = hf[ch][start:end]
                x[i, :len(sig)] = sig
        return torch.from_numpy(x), stage, hdf5_path
