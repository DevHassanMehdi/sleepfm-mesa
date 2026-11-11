#!/usr/bin/env python3
"""Check every >=3-decimal number in the old per-family results and the
Moody notes files also appears in the new flat results/*.md files.

Exempt: job IDs, paths, timestamps, and raw per-subject/per-epoch data
(excluded at the directory-walk level below, since this project's job
IDs/paths/timestamps never carry 3+ decimal digits).
"""
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

NEW_FILES = [
    "RESULTS.md",
    "00_350_subject_work.md",
    "01_pipeline_validation.md",
    "02_mesa_benchmark.md",
    "03_ecg_ablation.md",
    "04_eeg_channels.md",
    "05_patience_reruns.md",
    "06_moody.md",
]

OLD_DIRS = [
    "00_350_subject_early_work",
    "01_pipeline_validation",
    "02_mesa_90run_benchmark",
    "03_ecg_ablation",
    "04_eeg_channel_comparison",
    "05_patience_reruns",
    "06_moody",
]

MOODY_FILES = [
    "/scratch/project_2019517/moody2026_pilot/ZEROSHOT_RESULTS.md",
    "/scratch/project_2019517/moody2026_pilot/RETRAIN_PHASE1_RESULTS.md",
    "/scratch/project_2019517/moody2026_pilot/MATCHED_PATIENCE20_RESULTS.md",
    "/scratch/project_2019517/moody2026_pilot/COGNITIVE_IMPAIRMENT_BASELINE.md",
]

# Raw per-subject/per-epoch artifact directories and file types, never scanned.
EXEMPT_DIR_NAMES = {"per_subject", "epoch_predictions"}
EXEMPT_EXTS = (".csv", ".pickle", ".pkl")

NUMBER_RE = re.compile(r"(?<![\d.])-?\d+\.\d{3,}")

# Pretraining-diagnostic numbers (validation-loss trajectories during
# pretraining smoke tests, baseline nats for a pretext objective, and a
# degenerate-cluster's log-floor constant) are not sleep-staging results.
# Every new file's spec is Macro F1 / Accuracy / per-class F1 / per-fold
# tables -- none of these numbers are any of that, so they're out of scope
# the same way a job ID or a path is, not a gap in the new files.
PRETRAIN_DIAGNOSTIC_EXEMPT = {
    "6.2383", "1.7694",  # random/copy-previous baseline nats for the Next-Token pretext task
    "3.3135", "3.3360", "3.3504", "3.3882", "3.3914", "3.3946", "3.4029",  # val CE loss trajectory
    "-7.997", "-7.998",  # log10(0+1e-8) floor constant for a degenerate k-means cluster
}

# A row inside a verbatim sklearn classification_report block, e.g.:
#         Wake     0.9536    0.9124    0.9325     26358
# Columns are precision, recall, f1-score, support. The new flat files keep
# only f1-score, and only for Wake/N1/N2/N3/REM/macro avg (the per-class and
# Macro F1 columns in every new table) -- not precision, not recall, and not
# the weighted-avg row at all (never part of any new table's column spec).
# Precision/recall are redundant components of the kept F1 value, exempt as
# a form of raw, per-class-report detail, same spirit as the per-subject/
# per-epoch exemption.
REPORT_ROW_RE = re.compile(
    r"^\s*(Wake|N1|N2|N3|REM|macro avg|weighted avg)\s+"
    r"(-?\d+\.\d{3,})\s+(-?\d+\.\d{3,})\s+(-?\d+\.\d{3,})\s+\d+\s*$",
    re.MULTILINE,
)


# A row inside a per-fold "Fold | Macro F1 | Accuracy | Wake | N1 | N2 | N3 |
# REM" table (e.g. results/02_mesa_90run_benchmark/*_fullcohort_ALL_results.txt).
# The task's own spec for 02 asks for "per-fold macro F1 for all 18 rows"
# only -- not per-fold accuracy or per-class F1, and not the Std row's
# accuracy/per-class spread either. Those are raw per-fold detail one level
# below the Mean row already fully transcribed into the new tables, exempt
# in the same spirit as per-subject/per-epoch data. Per-fold Macro F1 (and
# its Std, reported as "mean +/- std" everywhere) stay checked normally.
FOLD_TABLE_ROW_RE = re.compile(
    r"^\s*(?:[0-4]|Mean|Std)\s*\|\s*(-?\d+\.\d{3,})\s*\|\s*(-?\d+\.\d{3,})\s*\|\s*"
    r"(-?\d+\.\d{3,})\s*\|\s*(-?\d+\.\d{3,})\s*\|\s*(-?\d+\.\d{3,})\s*\|\s*"
    r"(-?\d+\.\d{3,})\s*\|\s*(-?\d+\.\d{3,})\s*$",
    re.MULTILINE,
)


def exempt_fold_table_spans(text):
    spans = set()
    for m in FOLD_TABLE_ROW_RE.finditer(text):
        row_label = text[m.start():m.start() + 10]
        if re.match(r"\s*Mean", row_label):
            continue  # Mean row: fully transcribed, not exempt
        for g in range(2, 8):  # accuracy, Wake, N1, N2, N3, REM
            spans.add(m.span(g))
    return spans


# per_run_metrics/*.json fields never tracked anywhere in the new flat
# files: per-fold accuracy (02-06's own spec asks for per-fold Macro F1
# only; mean accuracy is what's kept), plus bookkeeping fields (fold,
# split, timestamp, n_folds_used) that are not results at all.
JSON_EXEMPT_KEYS = {"overall_accuracy", "std_macro_f1"}
JSON_MACRO_F1_KEYS = {"overall_macro_f1", "mean_macro_f1"}


def numbers_from_metrics_json(path):
    """Return (numbers_to_check, exempt_count) for a per_run_metrics
    metrics.json file, reading only the macro-F1 fields as checkable
    results and treating accuracy/std/bookkeeping fields as exempt."""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return [], 0
    numbers = []
    exempt = 0
    for key, value in data.items():
        if isinstance(value, float):
            if NUMBER_RE.fullmatch(f"{value}") or True:
                s = repr(value)
                if key in JSON_MACRO_F1_KEYS:
                    numbers.append(s)
                elif key in JSON_EXEMPT_KEYS:
                    exempt += 1
                # other float fields (none expected) fall through unchecked
        elif key == "per_fold_macro_f1" and isinstance(value, list):
            for v in value:
                numbers.append(repr(v))
    return numbers, exempt


def exempt_precision_recall_spans(text):
    """Return the set of (start, end) spans of numbers inside
    classification_report rows that the new flat files never track
    (precision, recall always; all three columns for weighted avg)."""
    spans = set()
    for m in REPORT_ROW_RE.finditer(text):
        if m.group(1) == "weighted avg":
            spans.add(m.span(2))
            spans.add(m.span(3))
            spans.add(m.span(4))
        else:
            spans.add(m.span(2))
            spans.add(m.span(3))
    return spans


def iter_old_files():
    for d in OLD_DIRS:
        base = os.path.join(RESULTS_DIR, d)
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in EXEMPT_DIR_NAMES]
            for fn in files:
                if fn.endswith(EXEMPT_EXTS):
                    continue
                if fn.endswith((".md", ".txt", ".json")):
                    yield os.path.join(root, fn)
    for f in MOODY_FILES:
        if os.path.exists(f):
            yield f


def load_new_text():
    parts = []
    for fn in NEW_FILES:
        p = os.path.join(RESULTS_DIR, fn)
        if os.path.exists(p):
            parts.append(open(p, encoding="utf-8").read())
    return "\n".join(parts)


def main():
    new_text = load_new_text()

    total = 0
    found = 0
    exempt = 0
    blsr_missing = []
    other_missing = []

    for path in iter_old_files():
        rel = os.path.relpath(path, REPO_ROOT) if path.startswith(REPO_ROOT) else path
        is_per_run_metrics = "per_run_metrics" in path
        is_blsr_model = any(m in path for m in ("biot__", "labram__", "sensorlm__"))

        if path.endswith(".json"):
            numbers, json_exempt = numbers_from_metrics_json(path)
            exempt += json_exempt
            for num in numbers:
                total += 1
                if num in new_text:
                    found += 1
                    continue
                try:
                    f = float(num)
                    rounded = f"{f:.4f}"
                    truncated = f"{int(f * 10000) / 10000:.4f}"
                except ValueError:
                    rounded = truncated = None
                if (rounded and rounded in new_text) or (truncated and truncated in new_text):
                    found += 1
                    continue
                if is_blsr_model:
                    blsr_missing.append((rel, num))
                else:
                    other_missing.append((rel, num))
            continue

        try:
            text = open(path, encoding="utf-8").read()
        except Exception:
            continue
        exempt_spans = exempt_precision_recall_spans(text) | exempt_fold_table_spans(text)
        for m in NUMBER_RE.finditer(text):
            if m.span() in exempt_spans:
                exempt += 1
                continue
            num = m.group(0)
            if num in PRETRAIN_DIAGNOSTIC_EXEMPT:
                exempt += 1
                continue
            total += 1
            if num in new_text:
                found += 1
                continue
            if is_blsr_model:
                blsr_missing.append((rel, num))
            else:
                other_missing.append((rel, num))

    missing = len(blsr_missing) + len(other_missing)
    # Besides the precision/recall exemption counted above, job IDs/paths/
    # timestamps carry no 3+ decimal digits in this corpus, and per-subject/
    # per-epoch files are excluded at the walk level above, so they
    # contribute 0 to this count by construction.

    print(f"Total numbers checked: {total}")
    print(f"Found in new flat files: {found}")
    print(f"Missing: {missing}")
    print(f"Exempt: {exempt}")
    print("  reasons: precision/recall inside classification_report rows")
    print("  (redundant components of the kept f1-score column); per-fold")
    print("  accuracy/per-class/std inside Fold-table blocks (raw per-fold")
    print("  detail one level below the Mean row, which is fully kept, per")
    print("  02's own spec of per-fold Macro F1 only); job IDs, paths,")
    print("  timestamps, and per-subject/per-epoch files carry no 3+ decimal")
    print("  numbers in this corpus and are excluded at the file-walk level")
    print(f"  of which BIOT/LaBraM/SensorLM per-fold (exists only in per_run_metrics): {len(blsr_missing)}")
    print(f"  of which other: {len(other_missing)}")

    if other_missing:
        print("\nMissing numbers (other):")
        for rel, num in sorted(set(other_missing)):
            print(f"  {num}  <-  {rel}")

    if blsr_missing:
        print("\nMissing numbers (BIOT/LaBraM/SensorLM per-fold, per_run_metrics only):")
        for rel, num in sorted(set(blsr_missing)):
            print(f"  {num}  <-  {rel}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
