"""Convert Moody/PhysioNet 2026 (site I0006) physiological_data EDFs into
MESA-layout HDF5 files: EEG1/EEG2/EEG3/EKG, float16, gzip, 5-min chunks,
resampled to 128 Hz.

EEG1 = F4 - M1, EEG2 = O2 - M1, EEG3 = C4 - M1, computed at the EDF's
native 200 Hz before resampling. EEG3 (C4-M1) is an exact electrode
match for MESA's own EEG3 (C4(+)-M1(-)). EEG1 and EEG2 are NOT electrode
matches for MESA's EEG1 (Fz-Cz) / EEG2 (Cz-Oz) -- I0006 has no Fz, Cz,
or Oz electrode at all. F4-M1 and O2-M1 are only approximations that
play a similar frontal/posterior role, using the same mastoid-reference
scheme I0006 actually has (and the standard AASM-recommended montage
for this electrode set).

Reuses sleepfm.preprocessing.preprocessing.EDFToHDF5Converter's
resample/standardize/save logic unmodified (same MESA HDF5 convention)
-- this script does not change that shared module's behavior or touch
any MESA checkpoint/data.

Usage:
    python scripts/moody_to_hdf5.py --edf_dir <physiological_data dir> \\
        --target_dir <output dir> [--num_files N]
"""
import argparse
import glob
import multiprocessing
import os
import sys
import time

# Moody/PhysioNet 2026 data is BDSP-licensed and must not be shared --
# every file this script creates (HDF5 outputs) must be owner-only.
os.umask(0o077)

import pyedflib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sleepfm"))

from preprocessing.preprocessing import EDFToHDF5Converter

REQUIRED_CHANNELS = ["F4", "O2", "C4", "M1", "EKG"]
# (output_name, positive_channel, negative_channel_or_None)
DERIVATIONS = [
    ("EEG1", "F4", "M1"),
    ("EEG2", "O2", "M1"),
    ("EEG3", "C4", "M1"),
    ("EKG", "EKG", None),
]
TARGET_RATE = 128


def read_required_channels(edf_path):
    """Returns {channel: (signal, sample_rate, physical_dimension)} for
    the 5 required channels, or raises ValueError describing what's
    missing/inconsistent."""
    f = pyedflib.EdfReader(edf_path)
    try:
        labels = f.getSignalLabels()
        missing = [ch for ch in REQUIRED_CHANNELS if ch not in labels]
        if missing:
            raise ValueError(f"missing channel(s): {missing}")

        out = {}
        for ch in REQUIRED_CHANNELS:
            idx = labels.index(ch)
            signal = f.readSignal(idx)
            rate = f.getSampleFrequency(idx)
            dim = f.getPhysicalDimension(idx)
            out[ch] = (signal, rate, dim)
        return out
    finally:
        f.close()


def check_pair_consistency(channels, ch_a, ch_b):
    """Equal sample rate, equal length, and equal physical unit --
    required before subtracting ch_a - ch_b."""
    sig_a, rate_a, dim_a = channels[ch_a]
    sig_b, rate_b, dim_b = channels[ch_b]
    if rate_a != rate_b:
        raise ValueError(f"{ch_a} ({rate_a} Hz) and {ch_b} ({rate_b} Hz) sample-rate mismatch")
    if len(sig_a) != len(sig_b):
        raise ValueError(f"{ch_a} ({len(sig_a)} samples) and {ch_b} ({len(sig_b)} samples) length mismatch")
    if dim_a != dim_b:
        raise ValueError(f"{ch_a} ({dim_a!r}) and {ch_b} ({dim_b!r}) physical-unit mismatch")


def convert_one(edf_path, target_dir, converter):
    subject_id = os.path.basename(edf_path).replace(".edf", "")
    hdf5_path = os.path.join(target_dir, f"{subject_id}.hdf5")

    channels = read_required_channels(edf_path)

    for _, pos, neg in DERIVATIONS:
        if neg is not None:
            check_pair_consistency(channels, pos, neg)

    derived_signals = []
    derived_names = []
    derived_rates = []
    for name, pos, neg in DERIVATIONS:
        sig_pos, rate, _ = channels[pos]
        if neg is not None:
            sig_neg, _, _ = channels[neg]
            derived = sig_pos - sig_neg
        else:
            derived = sig_pos
        derived_signals.append(derived)
        derived_names.append(name)
        derived_rates.append(rate)

    resampled = converter.resample_signals(derived_signals, derived_rates)
    converter.save_to_hdf5(resampled, derived_names, [], [], hdf5_path)
    return hdf5_path


_worker_converter = None


def _worker_init(target_dir):
    # Runs once per worker process (Pool initializer), not once per
    # file -- avoids EDFToHDF5Converter.__init__'s unused-but-not-free
    # recursive glob of the whole EDF directory on every conversion.
    global _worker_converter
    _worker_converter = EDFToHDF5Converter(root_dir=target_dir, target_dir=target_dir,
                                            resample_rate=TARGET_RATE)


def _worker(args):
    edf_path, target_dir = args
    subject_id = os.path.basename(edf_path).replace(".edf", "")
    t0 = time.time()
    try:
        out_path = convert_one(edf_path, target_dir, _worker_converter)
        return (subject_id, "OK", out_path, time.time() - t0)
    except ValueError as e:
        return (subject_id, "SKIP", str(e), time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_dir", type=str, required=True,
                         help="Directory containing I0006 physiological_data EDFs.")
    parser.add_argument("--target_dir", type=str, required=True,
                         help="Output directory for converted HDF5 files.")
    parser.add_argument("--file_list", type=str, default=None,
                         help="Optional text file of specific EDF filenames "
                              "(one per line, relative to --edf_dir) to "
                              "convert instead of every .edf in --edf_dir.")
    parser.add_argument("--num_files", type=int, default=-1,
                         help="Limit to the first N files found (-1 = all).")
    parser.add_argument("--num_workers", type=int, default=1,
                         help="Parallel worker processes (1 = sequential, "
                              "same as before this option existed).")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    if args.file_list:
        with open(args.file_list) as f:
            edf_files = [os.path.join(args.edf_dir, line.strip())
                         for line in f if line.strip()]
    else:
        edf_files = sorted(glob.glob(os.path.join(args.edf_dir, "*.edf")))

    if args.num_files != -1:
        edf_files = edf_files[:args.num_files]

    n_ok, n_skipped = 0, 0
    per_file_seconds = []

    if args.num_workers > 1:
        tasks = [(edf_path, args.target_dir) for edf_path in edf_files]
        with multiprocessing.Pool(args.num_workers, initializer=_worker_init,
                                   initargs=(args.target_dir,)) as pool:
            for subject_id, status, info, elapsed in pool.imap_unordered(_worker, tasks):
                if status == "OK":
                    per_file_seconds.append(elapsed)
                    print(f"OK ({elapsed:.1f}s): {subject_id} -> {info}", flush=True)
                    n_ok += 1
                else:
                    print(f"SKIP: {subject_id}: {info}", flush=True)
                    n_skipped += 1
    else:
        converter = EDFToHDF5Converter(root_dir=args.edf_dir, target_dir=args.target_dir,
                                        resample_rate=TARGET_RATE)
        for edf_path in edf_files:
            subject_id = os.path.basename(edf_path).replace(".edf", "")
            t0 = time.time()
            try:
                out_path = convert_one(edf_path, args.target_dir, converter)
                elapsed = time.time() - t0
                per_file_seconds.append(elapsed)
                print(f"OK ({elapsed:.1f}s): {subject_id} -> {out_path}", flush=True)
                n_ok += 1
            except ValueError as e:
                print(f"SKIP: {subject_id}: {e}", flush=True)
                n_skipped += 1

    print(f"\nDone. ok={n_ok} skipped={n_skipped} total={len(edf_files)}", flush=True)
    if per_file_seconds:
        print(f"mean conversion time: {sum(per_file_seconds) / len(per_file_seconds):.1f}s/file", flush=True)


if __name__ == "__main__":
    main()
