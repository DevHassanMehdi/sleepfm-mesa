"""Download SHHS PSG EDF + annotation files from NSRR (both visits).

Usage:
    python scripts/download_shhs.py --subjects 9200   # ~all V1 + all V2
    python scripts/download_shhs.py --subjects 2      # smoke test
"""

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NSRR_BIN = "/users/hamehdi/.local/share/x86_64/gem/ruby/3.4.0/bin/nsrr"
NSRR_RUBY_BINDIR = "/scratch/project_2019517/miniconda3/bin"
NSRR_GEM_ENV = {
    "PATH": NSRR_RUBY_BINDIR + os.pathsep + os.environ.get("PATH", ""),
}

EDF_MIN_BYTES = 20 * 1024 * 1024

# ID ranges (inclusive); not all IDs in range exist on NSRR
SHHS_V1_ID_START = 200001
SHHS_V1_ID_END   = 206441
SHHS_V2_ID_START = 200077
SHHS_V2_ID_END   = 205804

STAGE_MAP = {
    "wake": 0, "w": 0,
    "stage 1": 1, "n1": 1, "stage 1 sleep": 1,
    "stage 2": 2, "n2": 2, "stage 2 sleep": 2,
    "stage 3": 3, "n3": 3, "sws": 3, "stage 4": 3,
    "stage 3 sleep": 3, "stage 4 sleep": 3,
    "rem": 4, "stage r": 4, "stage r sleep": 4, "stage rem": 4, "rem sleep": 4,
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


def check_nsrr_binary() -> None:
    if not os.access(NSRR_BIN, os.X_OK):
        print(f"[ERROR] nsrr binary not found or not executable: {NSRR_BIN}")
        print("        Install it with: gem install nsrr --no-document")
        sys.exit(1)


def parse_xml(xml_path: Path) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for event in root.iter("ScoredEvent"):
        type_el    = event.find("EventType")
        concept_el = event.find("EventConcept")
        start_el   = event.find("Start")
        dur_el     = event.find("Duration")
        if type_el is None or concept_el is None or start_el is None or dur_el is None:
            continue
        if "Stages" not in (type_el.text or ""):
            continue
        raw_concept = (concept_el.text or "").strip()
        stage_name  = raw_concept.split("|")[0].strip()
        stage_num   = STAGE_MAP.get(stage_name.lower(), -1)
        start = float(start_el.text)
        rows.append({
            "Start": start,
            "Stop":  start + float(dur_el.text),
            "StageName":   stage_name,
            "StageNumber": stage_num,
        })
    return [r for r in rows if r["StageNumber"] >= 0]


def save_label_csv(rows: list, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Start,Stop,StageName,StageNumber\n")
        for r in rows:
            f.write(f"{r['Start']},{r['Stop']},{r['StageName']},{r['StageNumber']}\n")


def nsrr_download(remote_path: str, token: str, dest_path: Path) -> bool:
    env = {**os.environ, **NSRR_GEM_ENV}
    cmd = [NSRR_BIN, "download", remote_path, f"--token={token}"]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return False
    downloaded_path = REPO_ROOT / remote_path
    if not downloaded_path.exists():
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded_path), str(dest_path))
    return True


def scan_existing(visit: int) -> list:
    """Return sorted list of subject IDs with valid EDF + annotation already on disk."""
    prefix    = f"shhs{visit}"
    edf_dir   = REPO_ROOT / f"data/shhs/edf/{prefix}"
    annot_dir = REPO_ROOT / f"data/shhs/annotations-events-nsrr/{prefix}"
    if not edf_dir.exists():
        return []
    ids = []
    for edf_path in sorted(edf_dir.glob(f"{prefix}-*.edf")):
        if edf_path.stat().st_size < EDF_MIN_BYTES:
            continue
        # "shhs1-200001.edf" → "200001"
        sid = edf_path.stem[len(prefix) + 1:]
        if not sid.isdigit():
            continue
        if (annot_dir / f"{prefix}-{sid}-nsrr.xml").exists():
            ids.append(sid)
    return ids


def process_subject(sid: str, visit: int, token: str) -> tuple:
    """Download (or reuse) one subject for the given visit.

    Returns (status, label_generated) where status is "skipped", "downloaded",
    or "not_found".
    """
    prefix     = f"shhs{visit}"
    edf_path   = REPO_ROOT / f"data/shhs/edf/{prefix}/{prefix}-{sid}.edf"
    annot_path = REPO_ROOT / f"data/shhs/annotations-events-nsrr/{prefix}/{prefix}-{sid}-nsrr.xml"
    label_path = REPO_ROOT / f"data/shhs/labels/{prefix}/{prefix}-{sid}.csv"

    edf_ok   = edf_path.exists() and edf_path.stat().st_size >= EDF_MIN_BYTES
    annot_ok = annot_path.exists()
    label_generated = False

    if edf_ok and annot_ok:
        if not label_path.exists():
            try:
                save_label_csv(parse_xml(annot_path), label_path)
                label_generated = True
            except ET.ParseError as e:
                print(f"[ERROR] XML parse failed for {prefix}-{sid}: {e}")
        return "skipped", label_generated

    print(f"[INFO] Downloading {prefix}-{sid}...")

    edf_remote   = f"shhs/polysomnography/edfs/{prefix}/{prefix}-{sid}.edf"
    annot_remote = f"shhs/polysomnography/annotations-events-nsrr/{prefix}/{prefix}-{sid}-nsrr.xml"

    if not nsrr_download(edf_remote, token, edf_path):
        return "not_found", label_generated

    if edf_path.stat().st_size < EDF_MIN_BYTES:
        print(f"[WARN] EDF for {prefix}-{sid} is too small, removing")
        edf_path.unlink()
        return "not_found", label_generated

    if not nsrr_download(annot_remote, token, annot_path):
        print(f"[WARN] Annotation not found for {prefix}-{sid}")
        edf_path.unlink()
        return "not_found", label_generated

    try:
        save_label_csv(parse_xml(annot_path), label_path)
        label_generated = True
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed for {prefix}-{sid}: {e}")

    return "downloaded", label_generated


def main():
    parser = argparse.ArgumentParser(
        description="Download SHHS PSG data from NSRR (visit 1 then visit 2)"
    )
    parser.add_argument(
        "--subjects", type=int, required=True,
        help="Target total subjects across both visits (visit 1 downloaded first)"
    )
    args = parser.parse_args()

    check_nsrr_binary()
    token = load_token()

    for visit in [1, 2]:
        prefix = f"shhs{visit}"
        (REPO_ROOT / f"data/shhs/edf/{prefix}").mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / f"data/shhs/annotations-events-nsrr/{prefix}").mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / f"data/shhs/labels/{prefix}").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    pre_v1 = scan_existing(1)
    pre_v2 = scan_existing(2)
    total_existing = len(pre_v1) + len(pre_v2)

    print(f"  Already downloaded: {total_existing} subjects total")
    print(f"    Visit 1 (shhs1): {len(pre_v1)}"
          + (f"  (highest ID: {max(pre_v1)})" if pre_v1 else ""))
    print(f"    Visit 2 (shhs2): {len(pre_v2)}"
          + (f"  (highest ID: {max(pre_v2)})" if pre_v2 else ""))
    print(f"  Target: {args.subjects} total")

    target           = args.subjects
    completed        = total_existing
    downloaded       = 0
    not_found        = 0
    labels_generated = 0

    for visit, id_start, id_end, pre_existing in [
        (1, SHHS_V1_ID_START, SHHS_V1_ID_END, pre_v1),
        (2, SHHS_V2_ID_START, SHHS_V2_ID_END, pre_v2),
    ]:
        if completed >= target:
            break

        prefix = f"shhs{visit}"
        resume_from = (max(int(s) for s in pre_existing) + 1) if pre_existing else id_start

        if resume_from > id_end:
            print(f"  Visit {visit} ({prefix}): {len(pre_existing)} subjects already on disk, skipping")
            continue
        print(f"  Visit {visit} ({prefix}): resuming from ID {resume_from}")

        sid_num = resume_from
        while completed < target and sid_num <= id_end:
            sid = str(sid_num)
            status, lgen = process_subject(sid, visit, token)
            if lgen:
                labels_generated += 1
            if status == "downloaded":
                downloaded += 1
                completed += 1
            elif status == "skipped":
                completed += 1
            else:
                not_found += 1
            if status == "downloaded" or completed % 20 == 0:
                print(f"[INFO] Progress: {completed}/{target} | last: {prefix}-{sid} ({status})")
            sid_num += 1

    print("=" * 50)
    print(" SHHS Download Complete")
    print("=" * 50)
    print(f" Total      : {completed} subjects on disk")
    print(f" Downloaded : {downloaded} new in this run")
    print(f" Not found  : {not_found} IDs missing from NSRR")
    print(f" Labels     : {labels_generated} CSV files generated")
    print("=" * 50)


if __name__ == "__main__":
    main()
