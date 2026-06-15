"""Regenerate MESA label CSVs from already-downloaded annotation XML files.

Usage:
    python scripts/generate_labels.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ANNOT_DIR = REPO_ROOT / "data/mesa/annotations"
LABEL_DIR = REPO_ROOT / "data/mesa/labels"

STAGE_MAP = {
    "wake": 0, "w": 0,
    "stage 1": 1, "n1": 1, "stage 1 sleep": 1,
    "stage 2": 2, "n2": 2, "stage 2 sleep": 2,
    "stage 3": 3, "n3": 3, "sws": 3, "stage 4": 3,
    "stage 3 sleep": 3, "stage 4 sleep": 3,
    "rem": 4, "stage r": 4, "stage r sleep": 4, "stage rem": 4, "rem sleep": 4,
}


def parse_mesa_xml(xml_path: Path) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for event in root.iter("ScoredEvent"):
        type_el = event.find("EventType")
        concept_el = event.find("EventConcept")
        start_el = event.find("Start")
        dur_el = event.find("Duration")
        if type_el is None or concept_el is None or start_el is None or dur_el is None:
            continue
        event_type = (type_el.text or "").strip()
        if "Stages" not in event_type:
            continue
        raw_concept = (concept_el.text or "").strip()
        stage_name = raw_concept.split("|")[0].strip()
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


def main():
    if not ANNOT_DIR.exists():
        print(f"[ERROR] Annotations directory not found: {ANNOT_DIR}")
        return

    xml_files = sorted(ANNOT_DIR.glob("*.xml"))
    print(f"[INFO] Found {len(xml_files)} annotation files in {ANNOT_DIR}")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed = 0

    for xml_path in xml_files:
        subject_id = xml_path.name.split("-nsrr")[0]
        csv_path = LABEL_DIR / f"{subject_id}.csv"
        try:
            rows = parse_mesa_xml(xml_path)
            save_label_csv(rows, csv_path)
            generated += 1
            print(f"[OK] {subject_id}: {len(rows)} stage events -> {csv_path.name}")
        except ET.ParseError as e:
            print(f"[ERROR] Failed to parse {xml_path.name}: {e}")
            failed += 1

    print("=" * 46)
    print(" Label Generation Complete")
    print("=" * 46)
    print(f" Generated : {generated} CSV files")
    print(f" Failed    : {failed}")
    print(f" Output    : {LABEL_DIR}")
    print("=" * 46)


if __name__ == "__main__":
    main()
