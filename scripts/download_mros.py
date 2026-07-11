"""Download MrOS PSG EDF + annotation files from NSRR (both visits).

Usage:
    python scripts/download_mros.py --subjects 6600   # ~all V1 + all V2
    python scripts/download_mros.py --subjects 2      # smoke test
"""

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

NSRR_BIN = "/scratch/project_2019517/miniconda3/share/rubygems/bin/nsrr"
NSRR_GEM_ENV = {
    "GEM_HOME": "/scratch/project_2019517/miniconda3/share/rubygems",
    "GEM_PATH": "/scratch/project_2019517/miniconda3/share/rubygems",
}

EDF_MIN_BYTES = 50 * 1024 * 1024

# All known site prefixes (2-letter codes).  IDs are {prefix}{num:04d}.
MROS_PREFIXES = ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah",
                 "ai", "aj", "ak", "al", "am", "an", "ao", "ap"]
# Upper bound on subject number within a single prefix.  Real counts are
# much lower (~200–400 per prefix) so the loop exits at target, not here.
MROS_MAX_NUM = 9999

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
    """Return sorted list of subject IDs with valid EDF + annotation on disk."""
    visit_str = f"visit{visit}"
    edf_dir   = REPO_ROOT / f"data/mros/edf/{visit_str}"
    annot_dir = REPO_ROOT / f"data/mros/annotations-events-nsrr/{visit_str}"
    if not edf_dir.exists():
        return []
    prefix_stem = f"mros-{visit_str}-"
    ids = []
    for edf_path in sorted(edf_dir.glob(f"{prefix_stem}*.edf")):
        if edf_path.stat().st_size < EDF_MIN_BYTES:
            continue
        # "mros-visit1-aa0001.edf" → "aa0001"
        sid = edf_path.stem[len(prefix_stem):]
        if len(sid) != 6 or not sid[:2].isalpha() or not sid[2:].isdigit():
            continue
        if (annot_dir / f"{prefix_stem}{sid}-nsrr.xml").exists():
            ids.append(sid)
    return ids


def _next_mros_id(sid: str) -> Optional[str]:
    """Return the ID that follows sid in iteration order, or None if exhausted."""
    prefix, num = sid[:2], int(sid[2:])
    if num < MROS_MAX_NUM:
        return f"{prefix}{num + 1:04d}"
    prefix_idx = MROS_PREFIXES.index(prefix)
    if prefix_idx + 1 < len(MROS_PREFIXES):
        return f"{MROS_PREFIXES[prefix_idx + 1]}0001"
    return None


def _iter_mros_ids(start_sid: str = "aa0001") -> Iterator[str]:
    """Yield all MrOS IDs in order starting from (and including) start_sid."""
    # Locate starting prefix and number
    start_prefix = start_sid[:2]
    start_num    = int(start_sid[2:])
    started      = False
    for prefix in MROS_PREFIXES:
        for num in range(1, MROS_MAX_NUM + 1):
            if not started:
                if prefix == start_prefix and num == start_num:
                    started = True
                else:
                    continue
            yield f"{prefix}{num:04d}"


def process_subject(sid: str, visit: int, token: str) -> tuple:
    """Download (or reuse) one subject for the given visit.

    Returns (status, label_generated) where status is "skipped", "downloaded",
    or "not_found".
    """
    visit_str  = f"visit{visit}"
    stem       = f"mros-{visit_str}-{sid}"
    edf_path   = REPO_ROOT / f"data/mros/edf/{visit_str}/{stem}.edf"
    annot_path = REPO_ROOT / f"data/mros/annotations-events-nsrr/{visit_str}/{stem}-nsrr.xml"
    label_path = REPO_ROOT / f"data/mros/labels/{visit_str}/{stem}.csv"

    edf_ok   = edf_path.exists() and edf_path.stat().st_size >= EDF_MIN_BYTES
    annot_ok = annot_path.exists()
    label_generated = False

    if edf_ok and annot_ok:
        if not label_path.exists():
            try:
                save_label_csv(parse_xml(annot_path), label_path)
                label_generated = True
            except ET.ParseError as e:
                print(f"[ERROR] XML parse failed for {stem}: {e}")
        return "skipped", label_generated

    print(f"[INFO] Downloading {stem}...")

    edf_remote   = f"mros/polysomnography/edfs/{visit_str}/{stem}.edf"
    annot_remote = f"mros/polysomnography/annotations-events-nsrr/{visit_str}/{stem}-nsrr.xml"

    if not nsrr_download(edf_remote, token, edf_path):
        return "not_found", label_generated

    if edf_path.stat().st_size < EDF_MIN_BYTES:
        print(f"[WARN] EDF for {stem} is too small, removing")
        edf_path.unlink()
        return "not_found", label_generated

    if not nsrr_download(annot_remote, token, annot_path):
        print(f"[WARN] Annotation not found for {stem}")
        return "not_found", label_generated

    try:
        save_label_csv(parse_xml(annot_path), label_path)
        label_generated = True
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed for {stem}: {e}")

    return "downloaded", label_generated


def main():
    parser = argparse.ArgumentParser(
        description="Download MrOS PSG data from NSRR (visit 1 then visit 2)"
    )
    parser.add_argument(
        "--subjects", type=int, required=True,
        help="Target total subjects across both visits (visit 1 downloaded first)"
    )
    args = parser.parse_args()

    check_nsrr_binary()
    token = load_token()

    for visit in [1, 2]:
        visit_str = f"visit{visit}"
        (REPO_ROOT / f"data/mros/edf/{visit_str}").mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / f"data/mros/annotations-events-nsrr/{visit_str}").mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / f"data/mros/labels/{visit_str}").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    pre_v1 = scan_existing(1)
    pre_v2 = scan_existing(2)
    total_existing = len(pre_v1) + len(pre_v2)

    print(f"  Already downloaded: {total_existing} subjects total")
    print(f"    Visit 1: {len(pre_v1)}" + (f"  (highest ID: {max(pre_v1)})" if pre_v1 else ""))
    print(f"    Visit 2: {len(pre_v2)}" + (f"  (highest ID: {max(pre_v2)})" if pre_v2 else ""))
    print(f"  Target: {args.subjects} total")

    target           = args.subjects
    completed        = total_existing
    downloaded       = 0
    not_found        = 0
    labels_generated = 0

    for visit, pre_existing in [(1, pre_v1), (2, pre_v2)]:
        if completed >= target:
            break

        visit_str = f"visit{visit}"

        if pre_existing:
            start_sid = _next_mros_id(max(pre_existing))
        else:
            start_sid = "aa0001"

        if start_sid is None:
            print(f"  Visit {visit}: {len(pre_existing)} subjects already on disk, all IDs exhausted")
            continue
        print(f"  Visit {visit} (mros-{visit_str}): resuming from {start_sid}")

        for sid in _iter_mros_ids(start_sid):
            if completed >= target:
                break
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
                print(f"[INFO] Progress: {completed}/{target} | last: mros-{visit_str}-{sid} ({status})")

    print("=" * 50)
    print(" MrOS Download Complete")
    print("=" * 50)
    print(f" Total      : {completed} subjects on disk")
    print(f" Downloaded : {downloaded} new in this run")
    print(f" Not found  : {not_found} IDs missing from NSRR")
    print(f" Labels     : {labels_generated} CSV files generated")
    print("=" * 50)


if __name__ == "__main__":
    main()
