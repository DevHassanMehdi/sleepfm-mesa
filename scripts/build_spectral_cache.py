"""Build spectral reconstruction cache for pretrain_spectral.py.

Opens every HDF5 file ONCE, computes log10 spectral targets for every
640-sample window, and writes two files per split:

  {cache_dir}/spectral_signals_{split}.bin  — [N, max_ch, 640] float32
      Raw np.memmap — written/read page-by-page, never fully in RAM.

  {cache_dir}/spectral_cache_{split}.npz   — targets, masks, n_windows, n_files
      Small (~500 MB), loaded fully into RAM by SpectralDataset.

Run once on a small (CPU) node before launching pretrain_spectral.py:
    sbatch scripts/run_build_spectral_cache.slurm
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import tqdm
from loguru import logger
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sleepfm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import load_config, load_data


# EEG (EEG1/2/3) + EKG only -- narrowed from the earlier EEG-only scope to
# match the From-Scratch encoder's scope for a fair 3-way pretraining-method
# comparison, and deliberately NOT the full EEG+RESP+EKG+EMG scope (that
# would need MAX_CHANNELS=16 and was estimated at ~874GB / ~65h -- see
# PIPELINE_VALIDATION_FINDINGS.md Section 7).
MODALITY_TYPES = ["EEG_ONLY", "EKG"]
CHANNEL_GROUPS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sleepfm", "configs", "channel_groups.json"
)
with open(CHANNEL_GROUPS_PATH) as _f:
    CHANNEL_GROUPS = json.load(_f)

WINDOW_SIZE = 640
FS = 128
MAX_CHANNELS = 5  # EEG_ONLY(3) + EKG(2) -- matches config EEG_ONLY_CHANNELS/EKG_CHANNELS
BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 15.0), (15.0, 30.0)]


def compute_spectral_log10(signal):
    freqs, psd = welch(signal, fs=FS, nperseg=128)
    targets = np.zeros(5, dtype=np.float32)
    for j, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs < hi)
        power = float(psd[m].mean()) if m.any() else 0.0
        if np.isnan(power) or np.isinf(power):
            power = 0.0
        targets[j] = np.log10(power + 1e-8)
    return targets


def scan_windows(hdf5_paths):
    """First pass: count total windows and build (path, channels, start) index."""
    index = []
    skipped = 0
    for path in hdf5_paths:
        try:
            with h5py.File(path, "r") as hf:
                # Mirrors dataset.py's index_file_helper: scan the file's actual
                # dataset names and keep whichever belong to a target modality
                # group, rather than assuming a fixed channel-name list.
                available = [
                    ch for ch in hf.keys()
                    if any(ch in CHANNEL_GROUPS[g] for g in MODALITY_TYPES)
                ]
                if not available:
                    skipped += 1
                    continue
                n_samples = hf[available[0]].shape[0]
                n_windows = n_samples // WINDOW_SIZE
                for i in range(n_windows):
                    index.append((path, available, i * WINDOW_SIZE))
        except (OSError, AttributeError) as e:
            logger.warning(f"Skipping {path}: {e}")
            skipped += 1
    if skipped:
        logger.warning(f"Skipped {skipped} files (no EEG_ONLY/EKG channels or unreadable)")
    return index


def build_split_cache(split, config, cache_dir, val_size, suffix=""):
    all_names = load_data(config["split_path"])[split]
    hdf5_paths = [os.path.join(config["data_path"], p) for p in all_names]
    if split == "validation":
        hdf5_paths = hdf5_paths[:val_size]

    signals_path = os.path.join(cache_dir, f"spectral_signals_{split}{suffix}.bin")
    cache_path = os.path.join(cache_dir, f"spectral_cache_{split}{suffix}.npz")

    if os.path.isfile(signals_path) and os.path.isfile(cache_path):
        logger.info(f"[{split}] Cache already exists — skipping (delete files to rebuild)")
        return

    logger.info(f"[{split}] Scanning {len(hdf5_paths)} files for windows ...")
    index = scan_windows(hdf5_paths)
    N = len(index)
    if N == 0:
        logger.error(f"[{split}] No windows found — check data_path and split_path")
        return

    signals_gb = N * MAX_CHANNELS * WINDOW_SIZE * 4 / 1e9
    targets_mb = N * MAX_CHANNELS * 5 * 4 / 1e6
    logger.info(
        f"[{split}] {N} windows from {len(hdf5_paths)} files  "
        f"| signals: {signals_gb:.1f} GB (memmap)  "
        f"| targets: {targets_mb:.1f} MB (RAM)"
    )

    # Allocate signals as a raw memmap — written page-by-page, never fully in RAM
    os.makedirs(cache_dir, exist_ok=True)
    signals_mm = np.memmap(
        signals_path, dtype="float32", mode="w+",
        shape=(N, MAX_CHANNELS, WINDOW_SIZE),
    )

    targets = np.zeros((N, MAX_CHANNELS, 5), dtype=np.float32)
    masks = np.ones((N, MAX_CHANNELS), dtype=bool)  # True = padded

    for idx, (path, available, start) in enumerate(
        tqdm.tqdm(index, desc=f"[{split}]", unit="win")
    ):
        try:
            with h5py.File(path, "r") as hf:
                n_real = min(len(available), MAX_CHANNELS)
                for ci, ch in enumerate(available[:n_real]):
                    sig = hf[ch][start:start + WINDOW_SIZE].astype(np.float32)
                    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
                    signals_mm[idx, ci] = sig
                    targets[idx, ci] = compute_spectral_log10(sig)
                    masks[idx, ci] = False
        except Exception as e:
            logger.warning(f"  window {idx} ({path}:{start}): {e} — leaving zeros")

    del signals_mm  # flush and close the memmap

    real_mask = ~masks  # shape [N, MAX_CHANNELS], True = real channel
    all_t = targets[real_mask]
    logger.info(
        f"[{split}] Spectral targets: "
        f"mean={all_t.mean():.4f}  std={all_t.std():.4f}  "
        f"min={all_t.min():.4f}  max={all_t.max():.4f}"
    )

    # Save scan index so pretrain_combined.py can build a fast lookup without
    # re-opening HDF5 files.
    file_paths_arr = np.array([path.encode() for path, _, _ in index])
    window_starts_arr = np.array([start for _, _, start in index], dtype=np.int64)

    np.savez_compressed(
        cache_path,
        targets=targets,
        masks=masks,
        n_windows=np.array(N),
        n_files=np.array(len(hdf5_paths)),
        file_paths=file_paths_arr,
        window_starts=window_starts_arr,
    )

    logger.info(f"[{split}] Wrote {signals_path}  ({signals_gb:.1f} GB)")
    logger.info(f"[{split}] Wrote {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Build spectral cache for pretrain_spectral.py")
    parser.add_argument(
        "--config_path",
        default="sleepfm/configs/config_pretrain_spectral.yaml",
    )
    parser.add_argument(
        "--cache_dir",
        default="/scratch/project_2019517/sleepfm-data",
    )
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument(
        "--suffix",
        default="",
        help='String appended to output filenames before the extension, e.g. "_v2"',
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    for split in ["pretrain", "validation"]:
        build_split_cache(split, config, args.cache_dir, args.val_size, suffix=args.suffix)

    logger.info("Cache build complete.")


if __name__ == "__main__":
    main()
