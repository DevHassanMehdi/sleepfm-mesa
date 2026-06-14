"""Download MESA PSG EDF + annotation files from NSRR and generate label CSVs.

Usage:
    python scripts/download_mesa.py --subjects 350
    python scripts/download_mesa.py --subject-list data/mesa/subject_ids.txt
    python scripts/download_mesa.py --subjects 5
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EDF_DIR = REPO_ROOT / "data/mesa/edf"
ANNOT_DIR = REPO_ROOT / "data/mesa/annotations"
LABEL_DIR = REPO_ROOT / "data/mesa/labels"
LOGS_DIR = REPO_ROOT / "logs"
SPLIT_OUT = REPO_ROOT / "data/mesa/dataset_split_10fold.json"

NSRR_BIN = "/scratch/project_2019517/miniconda3/share/rubygems/bin/nsrr"
NSRR_GEM_ENV = {
    "GEM_HOME": "/scratch/project_2019517/miniconda3/share/rubygems",
    "GEM_PATH": "/scratch/project_2019517/miniconda3/share/rubygems",
}

MESA_EDF_REMOTE = "mesa/polysomnography/edfs"
MESA_ANNOT_REMOTE = "mesa/polysomnography/annotations-events-nsrr"
EDF_MIN_BYTES = 50 * 1024 * 1024
MAX_SUBJECT_ID = 9999

STAGE_MAP = {
    "wake": 0, "w": 0,
    "stage 1": 1, "n1": 1, "stage 1 sleep": 1,
    "stage 2": 2, "n2": 2, "stage 2 sleep": 2,
    "stage 3": 3, "n3": 3, "sws": 3, "stage 4": 3,
    "stage 3 sleep": 3, "stage 4 sleep": 3,
    "rem": 4, "stage r": 4, "stage rem": 4, "rem sleep": 4,
}


def load_token() -> str:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("NSRR_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                if token and token != "your_token_here":
                    return token
    print("[ERROR] NSRR_TOKEN not found in .env")
    print("        Create .env with:  NSRR_TOKEN=your_token_here")
    sys.exit(1)


def parse_mesa_xml(xml_path: Path) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for event in root.iter("ScoredEvent"):
        name_el = event.find("Name")
        start_el = event.find("Start")
        dur_el = event.find("Duration")
        if name_el is None or start_el is None or dur_el is None:
            continue
        raw_name = (name_el.text or "").strip()
        stage_name = raw_name.split("|")[0].strip()
        stage_num = STAGE_MAP.get(stage_name.lower(), -1)
        start = float(start_el.text)
        stop = start + float(dur_el.text)
        rows.append({
            "Start": start,
            "Stop": stop,
            "StageName": stage_name,
            "StageNumber": stage_num,
        })
    return [r for r in rows if r["StageNumber"] >= 0]


def save_label_csv(rows: list, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Start,Stop,StageName,StageNumber\n")
        for r in rows:
            f.write(f"{r['Start']},{r['Stop']},{r['StageName']},{r['StageNumber']}\n")


def check_nsrr_binary() -> None:
    if not os.access(NSRR_BIN, os.X_OK):
        print(f"[ERROR] nsrr binary not found or not executable: {NSRR_BIN}")
        print("        Install it with: gem install nsrr --no-document")
        sys.exit(1)


def nsrr_download(remote_path: str, token: str, dest_path: Path) -> bool:
    env = {**os.environ, **NSRR_GEM_ENV}
    cmd = [NSRR_BIN, "download", remote_path, f"--token={token}"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[ERROR] nsrr download failed for {remote_path}: {result.stderr.strip()}")
        return False

    downloaded_path = REPO_ROOT / remote_path
    if not downloaded_path.exists():
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded_path), str(dest_path))
    return True


def process_subject(sid: str, token: str):
    """Download (or reuse) a single subject's EDF/annotation/label files.

    Returns a tuple of (status, label_generated) where status is one of
    "skipped", "downloaded", or "not_found".
    """
    edf_path = EDF_DIR / f"mesa-sleep-{sid}.edf"
    annot_path = ANNOT_DIR / f"mesa-sleep-{sid}-nsrr.xml"
    label_path = LABEL_DIR / f"mesa-sleep-{sid}.csv"

    edf_ok = edf_path.exists() and edf_path.stat().st_size >= EDF_MIN_BYTES
    annot_ok = annot_path.exists()

    label_generated = False

    if edf_ok and annot_ok:
        if not label_path.exists():
            try:
                rows = parse_mesa_xml(annot_path)
                save_label_csv(rows, label_path)
                label_generated = True
            except ET.ParseError as e:
                print(f"[ERROR] Failed to parse XML for {sid}: {e}")
        return "skipped", label_generated

    edf_remote = f"{MESA_EDF_REMOTE}/mesa-sleep-{sid}.edf"
    annot_remote = f"{MESA_ANNOT_REMOTE}/mesa-sleep-{sid}-nsrr.xml"

    print(f"[INFO] Downloading subject {sid}...")

    edf_success = nsrr_download(edf_remote, token, edf_path)
    if not edf_success:
        print(f"[WARN] EDF not found for subject {sid}")
        return "not_found", label_generated

    if edf_path.stat().st_size < EDF_MIN_BYTES:
        print(f"[WARN] EDF for subject {sid} is too small, removing")
        edf_path.unlink()
        return "not_found", label_generated

    annot_success = nsrr_download(annot_remote, token, annot_path)
    if not annot_success:
        print(f"[WARN] Annotation not found for subject {sid}")
        return "not_found", label_generated

    try:
        rows = parse_mesa_xml(annot_path)
        save_label_csv(rows, label_path)
        label_generated = True
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse XML for {sid}: {e}")

    return "downloaded", label_generated


def scan_existing_subjects() -> list:
    """Find all subjects with a valid EDF + annotation already on disk.

    Regenerates any missing label CSVs from the annotation XML so the
    split stays consistent across resumed runs.
    """
    subject_ids = []
    for edf_path in sorted(EDF_DIR.glob("mesa-sleep-*.edf")):
        if edf_path.stat().st_size < EDF_MIN_BYTES:
            continue
        sid = edf_path.stem.split("-")[-1]
        annot_path = ANNOT_DIR / f"mesa-sleep-{sid}-nsrr.xml"
        label_path = LABEL_DIR / f"mesa-sleep-{sid}.csv"
        if not annot_path.exists():
            continue
        if not label_path.exists():
            try:
                rows = parse_mesa_xml(annot_path)
                save_label_csv(rows, label_path)
            except ET.ParseError as e:
                print(f"[ERROR] Failed to parse XML for {sid}: {e}")
        subject_ids.append(sid)
    return subject_ids


def generate_10fold_split(subject_ids: list, output_path: Path):
    random.seed(42)
    ids = subject_ids.copy()
    random.shuffle(ids)
    n = len(ids)
    fold_size = n // 10
    folds = [ids[i * fold_size:(i + 1) * fold_size] for i in range(10)]
    folds[-1].extend(ids[10 * fold_size:])
    split = {}
    for i in range(10):
        test_ids = folds[i]
        val_ids = folds[(i + 1) % 10]
        train_ids = []
        for j in range(10):
            if j != i and j != (i + 1) % 10:
                train_ids.extend(folds[j])
        split[f"fold_{i}"] = {
            "train": [f"mesa/mesa-sleep-{sid}.hdf5" for sid in train_ids],
            "validation": [f"mesa/mesa-sleep-{sid}.hdf5" for sid in val_ids],
            "test": [f"mesa/mesa-sleep-{sid}.hdf5" for sid in test_ids],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"  Saved 10-fold split to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download MESA PSG data from NSRR")
    parser.add_argument("--subjects", type=int, default=None,
                         help="Download N sequential subjects starting from 0001")
    parser.add_argument("--subject-list", type=str, default=None,
                         help="Path to text file with one 4-digit subject ID per line")
    args = parser.parse_args()

    if not args.subjects and not args.subject_list:
        parser.error("Must specify either --subjects or --subject-list")

    check_nsrr_binary()
    token = load_token()

    EDF_DIR.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    not_found = 0
    labels_generated = 0
    completed_ids = []

    if args.subject_list:
        subject_ids = []
        for line in Path(args.subject_list).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                subject_ids.append(line.zfill(4))

        for sid in subject_ids:
            if int(sid) > MAX_SUBJECT_ID:
                continue

            status, label_generated = process_subject(sid, token)
            if label_generated:
                labels_generated += 1

            if status == "skipped":
                skipped += 1
                completed_ids.append(sid)
            elif status == "downloaded":
                downloaded += 1
                completed_ids.append(sid)
            else:
                not_found += 1
    else:
        target = args.subjects
        sid_num = 1

        while len(completed_ids) < target and sid_num <= MAX_SUBJECT_ID:
            sid = str(sid_num).zfill(4)

            status, label_generated = process_subject(sid, token)
            if label_generated:
                labels_generated += 1

            if status == "skipped":
                skipped += 1
                completed_ids.append(sid)
            elif status == "downloaded":
                downloaded += 1
                completed_ids.append(sid)
            else:
                not_found += 1

            print(f"[INFO] Progress: {len(completed_ids)}/{target} subjects completed")
            sid_num += 1

    existing_ids = scan_existing_subjects()
    all_completed_ids = sorted(set(existing_ids) | set(completed_ids))
    if all_completed_ids:
        generate_10fold_split(all_completed_ids, SPLIT_OUT)

    print("=" * 46)
    print(" MESA Download Complete")
    print("=" * 46)
    print(f" Downloaded : {downloaded} subjects")
    print(f" Skipped    : {skipped} (already present)")
    print(f" Not found  : {not_found} (missing from NSRR)")
    print(f" Labels     : {labels_generated} CSV files generated")
    print(f" Split file : data/mesa/dataset_split_10fold.json")
    print("=" * 46)


if __name__ == "__main__":
    main()
