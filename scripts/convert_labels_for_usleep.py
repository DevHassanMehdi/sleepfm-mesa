#!/usr/bin/env python3
"""
Convert SleepFM-MESA per-epoch label CSVs (Start,Stop,StageName,StageNumber)
to U-Sleep's .ids hypnogram format: a headerless CSV of
(Start_sec, Duration_sec, StageLabel) where StageLabel matches the keys in
the dataset config's sleep_stage_annotations (W, N1, N2, N3, REM).

Usage:
    python scripts/convert_labels_for_usleep.py \
        --labels_dir data/mesa/labels \
        --out_dir /scratch/project_2019517/usleep_mesa/hypnograms
"""
import argparse
import csv
import os
from pathlib import Path

STAGE_NUMBER_TO_LABEL = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def convert_one(csv_path: Path, out_path: Path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = float(row["Start"])
            stop = float(row["Stop"])
            stage_number = int(row["StageNumber"])
            label = STAGE_NUMBER_TO_LABEL[stage_number]
            rows.append((start, stop - start, label))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", default="data/mesa/labels")
    parser.add_argument("--out_dir", default="/scratch/project_2019517/usleep_mesa/hypnograms")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(labels_dir.glob("mesa-sleep-*.csv"))
    print(f"Found {len(csv_files)} label files in {labels_dir}")

    converted = 0
    for csv_path in csv_files:
        subject_id = csv_path.stem  # e.g. mesa-sleep-0001
        out_path = out_dir / f"{subject_id}.ids"
        convert_one(csv_path, out_path)
        converted += 1

    print(f"Converted {converted} hypnograms to {out_dir}")


if __name__ == "__main__":
    main()
