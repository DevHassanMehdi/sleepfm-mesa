"""Convert Moody/PhysioNet 2026 (site I0006) human_annotations EDFs into
our label CSV format: Start, Stop, StageName, StageNumber -- one row per
30s epoch, every row kept (dropping rows would shift every later epoch's
alignment against the signal HDF5).

Stage-code mapping, I0006 (stage_expert channel) -> ours:
    5 (Wake) -> 0
    3 (N1)   -> 1
    2 (N2)   -> 2
    1 (N3)   -> 3
    4 (REM)  -> 4
    0        -> -1 (Unscored)
    9        -> -1 (Unscored; never actually observed in I0006, but
                     mapped the same way per the Challenge's own stated
                     encoding, "9 = Unavailable")

Code 0 is not in the Challenge's own documented encoding ("1=N3, 2=N2,
3=N1, 4=REM, 5=Wake, 9=Unavailable" -- physionetchallenges.org/2026/).
The official example repo (python-example-2026/team_code.py) is
internally inconsistent about it: one comment block claims
"0=W...5=Unknown/Movement" for stage_expert specifically, but the
executable code in that same function uses w_pct = mean(stage==5) --
i.e. 5=Wake, matching the official page, not the comment. Treated as
unscored here regardless (per instruction), consistent with how it
behaves in the real I0006 data: a small share in most records,
concentrated at ~100% in a handful of records that are otherwise
unscored end to end.

-1 is MESA's own existing "no label" sentinel (sleepfm/models/dataset.py
replaces it with 0 for the model's purposes, but a separate validity
mask -- added specifically to support this -- excludes it from the
loss and metrics before that remap happens).
"""
import argparse
import glob
import multiprocessing
import os
import sys

# Moody/PhysioNet 2026 data is BDSP-licensed and must not be shared --
# every file this script creates (label CSVs) must be owner-only.
os.umask(0o077)

import pyedflib
import pandas as pd

EPOCH_SEC = 30
STAGE_MAP = {5: 0, 3: 1, 2: 2, 1: 3, 4: 4, 0: -1, 9: -1}
STAGE_NAME = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "Unscored"}


def convert_one(annot_path, csv_path):
    f = pyedflib.EdfReader(annot_path)
    try:
        labels = f.getSignalLabels()
        idx = labels.index("stage_expert")
        raw_codes = f.readSignal(idx)
    finally:
        f.close()

    rows = []
    for i, raw_code in enumerate(raw_codes):
        raw_code = int(raw_code)
        if raw_code not in STAGE_MAP:
            raise ValueError(f"unexpected stage_expert code {raw_code} at epoch {i}")
        stage_number = STAGE_MAP[raw_code]
        rows.append({
            "Start": i * EPOCH_SEC,
            "Stop": (i + 1) * EPOCH_SEC,
            "StageName": STAGE_NAME[stage_number],
            "StageNumber": stage_number,
        })

    df = pd.DataFrame(rows, columns=["Start", "Stop", "StageName", "StageNumber"])
    df.to_csv(csv_path, index=False)
    return len(df)


def _worker(args):
    annot_path, target_dir = args
    subject_id = os.path.basename(annot_path).replace("_expert_annotations.edf", "")
    csv_path = os.path.join(target_dir, f"{subject_id}.csv")
    try:
        n_rows = convert_one(annot_path, csv_path)
        return (subject_id, "OK", f"{csv_path} ({n_rows} rows)")
    except Exception as e:
        return (subject_id, "FAILED", str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annot_dir", type=str, required=True,
                         help="Directory containing I0006 human_annotations EDFs.")
    parser.add_argument("--target_dir", type=str, required=True,
                         help="Output directory for converted label CSVs.")
    parser.add_argument("--file_list", type=str, default=None,
                         help="Optional text file of specific EDF filenames "
                              "(one per line, relative to --annot_dir) to "
                              "convert instead of every .edf in --annot_dir.")
    parser.add_argument("--num_files", type=int, default=-1,
                         help="Limit to the first N files found (-1 = all).")
    parser.add_argument("--num_workers", type=int, default=1,
                         help="Parallel worker processes (1 = sequential).")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    if args.file_list:
        with open(args.file_list) as f:
            annot_files = [os.path.join(args.annot_dir, line.strip())
                           for line in f if line.strip()]
    else:
        annot_files = sorted(glob.glob(os.path.join(args.annot_dir, "*.edf")))

    if args.num_files != -1:
        annot_files = annot_files[:args.num_files]

    n_ok, n_failed = 0, 0

    if args.num_workers > 1:
        tasks = [(annot_path, args.target_dir) for annot_path in annot_files]
        with multiprocessing.Pool(args.num_workers) as pool:
            for subject_id, status, info in pool.imap_unordered(_worker, tasks, chunksize=4):
                if status == "OK":
                    print(f"OK: {subject_id} -> {info}", flush=True)
                    n_ok += 1
                else:
                    print(f"FAILED: {subject_id}: {info}", flush=True)
                    n_failed += 1
    else:
        for annot_path in annot_files:
            subject_id = os.path.basename(annot_path).replace("_expert_annotations.edf", "")
            csv_path = os.path.join(args.target_dir, f"{subject_id}.csv")
            try:
                n_rows = convert_one(annot_path, csv_path)
                print(f"OK: {subject_id} -> {csv_path} ({n_rows} rows)", flush=True)
                n_ok += 1
            except Exception as e:
                print(f"FAILED: {subject_id}: {e}", flush=True)
                n_failed += 1

    print(f"\nDone. ok={n_ok} failed={n_failed} total={len(annot_files)}", flush=True)


if __name__ == "__main__":
    main()
