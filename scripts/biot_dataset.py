"""
PyTorch Dataset for fine-tuning BIOT on MESA PSG sleep staging.

Reads 30-second epochs straight out of the MESA HDF5 files (128Hz) and
resamples them to 200Hz to match BIOT's pretrained STFT parameters
(n_fft=200, hop=100), since the pretrained checkpoints were trained on
signals resampled to 200Hz (see BIOT/README.md).
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
BIOT_HZ = 200
EPOCH_SEC = 30
MESA_SAMPLES = MESA_HZ * EPOCH_SEC   # 3840
BIOT_SAMPLES = BIOT_HZ * EPOCH_SEC   # 6000

HDF5_DIR = "data/mesa/hdf5"
LABELS_DIR = "data/mesa/labels"

MODALITY_CHANNELS = {
    "EEG_ONLY":          ["EEG1", "EEG2", "EEG3"],
    "ECG_ONLY":          ["EKG"],
    "EEG_ECG":           ["EEG1", "EEG2", "EEG3", "EKG"],
    "BAS":               ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R"],
    "BAS_EKG":           ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R", "EKG"],
    "BAS_EKG_RESP":      ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R", "EKG",
                           "Abdo", "HR", "Snore", "SpO2", "Therm", "Thor"],
    "BAS_EKG_RESP_EMG":  ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R", "EKG",
                           "Abdo", "HR", "Snore", "SpO2", "Therm", "Thor",
                           "EMG", "Leg", "Pleth"],
}

_SPLIT_KEY = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def get_subject_id(filename):
    return filename.replace(".hdf5", "")


def _normalize(x):
    """BIOT's own preprocessing convention (see utils.py *Loader classes in
    the BIOT repo): scale each channel by its 95th percentile of |amplitude|
    so inputs sit on the same scale the pretrained weights were trained on."""
    q = np.quantile(np.abs(x), q=0.95, axis=-1, keepdims=True)
    return x / (q + 1e-8)


class BIOTSleepDataset(Dataset):
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
        x = resample(x, BIOT_SAMPLES, axis=-1)
        x = _normalize(x)
        x = torch.from_numpy(x.astype(np.float32))
        return x, stage
