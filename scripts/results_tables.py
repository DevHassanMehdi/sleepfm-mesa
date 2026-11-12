#!/usr/bin/env python3
"""Parse, check, and fix the bold/green-square marking on every markdown
table in results/*.md, driven by a spec comment directly above each
table.

Spec comment format (one line, directly above the table, nothing between
it and the table's header row):

    <!-- spec: block=<ColName|none>; items=<free text for the caption>;
    value_cols=<Col1,Col2,...>; bold=down|none; green=along|none;
    exclude_rows=<ColName>~<substring>[|<ColName>~<substring>...];
    tie=0.002; marks=none; note=<free text, only used when marks=none> -->

- block: the column whose value groups rows into blocks (ranking is done
  separately within each block). A blank cell in the block column
  inherits the previous row's value (carried down). "none" means the
  whole table is one block.
- items: free text describing what is being compared, used only to
  generate the caption.
- value_cols: columns ranked for bold (and, if green=along, also used
  for the along-row green check), by exact header text.
- bold: "down" marks, within each block, every cell in a value column
  that is the block's max in that column, or within `tie` of it.
- green: "along" marks, within each row, the value_cols cell(s) holding
  the row's max among those columns (or within `tie` of it). Requires at
  least two valid (non n/a, non "--") values in the row to apply.
- exclude_rows: rows whose named column's cell contains the given
  substring are skipped entirely: never marked, and not counted as
  competition for other rows.
- tie: tolerance for "also gets the mark" comparisons. Default 0.002.
- marks=none: this table gets no bold/green at all (still gets a spec
  and a caption).

Cell value parsing: strip `**`, strip the square emoji, take the first
decimal number in what remains. A cell that is "n/a", "--", "-", or
empty (after stripping) has no value and is skipped (not ranked, not
required to carry a mark).
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

GREEN = "\U0001F7E9"  # 🟩
BLUE = "\U0001F7E6"  # 🟦  (legacy marker, no longer part of the vocabulary)
NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")
SPEC_RE = re.compile(r"^<!--\s*spec:\s*(.*?)\s*-->\s*$")
CAPTION_RE = re.compile(r"^\*.*\*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def parse_spec(text):
    spec = {}
    for part in text.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        spec[k.strip()] = v.strip()
    return spec


def spec_to_text(spec):
    order = ["block", "items", "value_cols", "bold", "green", "exclude_rows", "tie", "marks", "note"]
    parts = []
    for k in order:
        if k in spec:
            parts.append(f"{k}={spec[k]}")
    return "spec: " + "; ".join(parts)


def make_caption(spec):
    if spec.get("marks") == "none":
        reason = spec.get("note", "not a head-to-head comparison")
        return f"*No marks: {reason}.*"
    items = spec.get("items", "items")
    block = spec.get("block", "none")
    if block and block.lower() != "none":
        scope = "down each column within each block"
    else:
        scope = "down each column"
    cap = f"*Compared: {items}, {scope}, higher is better."
    if spec.get("green") == "along":
        cap += " Green square: best dataset setting along each row."
    cap += "*"
    return cap


UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")


def split_row(line):
    """Split a markdown table row into cells, treating `\\|` as a literal
    pipe inside a cell (not a column delimiter)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in UNESCAPED_PIPE_RE.split(s)]


def join_row(cells):
    return "| " + " | ".join(cells) + " |"


def cell_value(cell):
    """Return the float value of a cell, or None if it has no value
    (n/a, --, -, empty) after stripping bold/square decoration."""
    s = cell.strip()
    s = s.replace("**", "")
    s = s.replace(GREEN, "").replace(BLUE, "")
    s = s.strip()
    if s in ("", "-", "--", "n/a") or s.lower().startswith("n/a"):
        return None
    m = NUMBER_RE.search(s)
    if m is None:
        return None
    return float(m.group(0))


def strip_decoration(cell):
    """Return the cell's text with existing bold/square marks removed,
    preserving the rest of its content (value, ± std, trailing notes)."""
    s = cell
    s = s.replace(GREEN, "").replace(BLUE, "")
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def has_bold(cell):
    return "**" in cell


def has_green(cell):
    return GREEN in cell


def has_blue(cell):
    return BLUE in cell


class Table:
    def __init__(self, file, header_line_idx, header, rows_start, rows_end, lines):
        self.file = file
        self.header_line_idx = header_line_idx  # index of the header row in `lines`
        self.header = header  # list of column names
        self.rows_start = rows_start  # index of first data row
        self.rows_end = rows_end  # index one past the last data row
        self.lines = lines  # the full file's lines (shared, mutated in place)
        self.spec_line_idx = None
        self.caption_line_idx = None
        self.spec = None

    def col_index(self, name):
        for i, h in enumerate(self.header):
            if h.strip() == name.strip():
                return i
        return None

    def data_rows(self):
        return list(range(self.rows_start, self.rows_end))


def find_tables(lines):
    """Find every markdown table in `lines`. Returns a list of Table."""
    tables = []
    i = 0
    n = len(lines)
    while i < n - 1:
        if lines[i].strip().startswith("|") and TABLE_SEP_RE.match(lines[i + 1]):
            header = split_row(lines[i])
            j = i + 2
            while j < n and lines[j].strip().startswith("|"):
                j += 1
            t = Table(None, i, header, i + 2, j, lines)
            # look for spec/caption directly above
            k = i - 1
            if k >= 0 and SPEC_RE.match(lines[k].strip()):
                t.spec_line_idx = k
                t.spec = parse_spec(SPEC_RE.match(lines[k].strip()).group(1))
                k -= 1
                if k >= 0 and CAPTION_RE.match(lines[k].strip()):
                    t.caption_line_idx = k
            tables.append(t)
            i = j
        else:
            i += 1
    return tables


def resolve_block_values(table, block_col_idx):
    """Return a list (one per data row) of the block value for that row,
    carrying the previous row's value down over blank cells."""
    values = []
    last = None
    for ridx in table.data_rows():
        cells = split_row(table.lines[ridx])
        v = cells[block_col_idx].strip() if block_col_idx is not None else ""
        if block_col_idx is not None and v == "":
            v = last
        else:
            last = v
        values.append(v)
    return values


def row_excluded(cells, header, exclude_rows_spec):
    if not exclude_rows_spec:
        return False
    for clause in exclude_rows_spec.split("|"):
        clause = clause.strip()
        if "~" not in clause:
            continue
        col, substr = clause.split("~", 1)
        idx = None
        for i, h in enumerate(header):
            if h.strip() == col.strip():
                idx = i
                break
        if idx is not None and idx < len(cells) and substr in cells[idx]:
            return True
    return False


def compute_marks(table):
    """Returns {row_idx: {col_idx: 'bold'|'green'|'both'}} for this table
    under its spec. Does not mutate the file."""
    spec = table.spec
    marks = {}
    if spec is None or spec.get("marks") == "none":
        return marks
    tie = float(spec.get("tie", 0.002))
    value_cols = [c.strip() for c in spec.get("value_cols", "").split(",") if c.strip()]
    value_col_idx = [table.col_index(c) for c in value_cols]
    if any(idx is None for idx in value_col_idx):
        missing = [c for c, idx in zip(value_cols, value_col_idx) if idx is None]
        raise ValueError(f"{table.file}: value_cols not found in header {table.header}: {missing}")

    block = spec.get("block", "none")
    block_col_idx = table.col_index(block) if block and block.lower() != "none" else None
    block_values = resolve_block_values(table, block_col_idx)

    exclude_spec = spec.get("exclude_rows", "")
    row_cells = {ridx: split_row(table.lines[ridx]) for ridx in table.data_rows()}
    row_is_excluded = {
        ridx: row_excluded(row_cells[ridx], table.header, exclude_spec)
        for ridx in table.data_rows()
    }

    # ---- bold: down each value column, within each block ----
    if spec.get("bold") == "down":
        for col_idx in value_col_idx:
            # group row indices by block value
            groups = {}
            for ridx, bval in zip(table.data_rows(), block_values):
                if row_is_excluded[ridx]:
                    continue
                groups.setdefault(bval, []).append(ridx)
            for bval, ridxs in groups.items():
                vals = []
                for ridx in ridxs:
                    v = cell_value(row_cells[ridx][col_idx])
                    if v is not None:
                        vals.append((ridx, v))
                if not vals:
                    continue
                best = max(v for _, v in vals)
                for ridx, v in vals:
                    if v >= best - tie:
                        marks.setdefault(ridx, {}).setdefault(col_idx, set()).add("bold")

    # ---- green: along each row, across value_cols ----
    if spec.get("green") == "along":
        for ridx in table.data_rows():
            if row_is_excluded[ridx]:
                continue
            cells = row_cells[ridx]
            vals = [(idx, cell_value(cells[idx])) for idx in value_col_idx]
            vals = [(idx, v) for idx, v in vals if v is not None]
            if len(vals) < 2:
                continue
            best = max(v for _, v in vals)
            for idx, v in vals:
                if v >= best - tie:
                    marks.setdefault(ridx, {}).setdefault(idx, set()).add("green")

    return marks


def apply_cell_mark(cell, mark_set):
    """Rebuild a cell's text with exactly the marks in mark_set (a subset
    of {'bold','green'}), preserving its underlying text."""
    text = strip_decoration(cell)
    if not mark_set:
        return text
    if "bold" in mark_set:
        text = f"**{text}**"
    if "green" in mark_set:
        text = f"{text} {GREEN}"
    return text


def rewrite_table_marks(table):
    """Mutate table.lines in place: apply computed marks to every data
    row's value cells, and strip marks from any other cell (excluded
    rows, non-value columns, or when marks=none)."""
    spec = table.spec
    value_cols = [c.strip() for c in spec.get("value_cols", "").split(",") if c.strip()] if spec else []
    value_col_idx = set(table.col_index(c) for c in value_cols)
    marks = compute_marks(table) if spec is not None else {}

    exclude_spec = spec.get("exclude_rows", "") if spec else ""
    for ridx in table.data_rows():
        cells = split_row(table.lines[ridx])
        excluded = row_excluded(cells, table.header, exclude_spec)
        new_cells = []
        for cidx, cell in enumerate(cells):
            if spec is not None and spec.get("marks") != "none" and cidx in value_col_idx and not excluded:
                mark_set = marks.get(ridx, {}).get(cidx, set())
                new_cells.append(apply_cell_mark(cell, mark_set))
            else:
                # not a ranked cell under this spec: strip any decoration
                new_cells.append(strip_decoration(cell))
        table.lines[ridx] = join_row(new_cells)


# Known, pre-existing rows whose five per-class F1 values do not average
# exactly to the stated macro F1 -- confirmed unchanged from their source
# (the BNAIC paper, for the SleepFM From-Scratch paper rows in
# 00_350_subject_work.md; the raw per-window predictions behind the other
# rows no longer exist either way), so the discrepancy can't be fixed
# without altering a historical number. Keyed by (row label, macro F1);
# the stored diff must match the one measured now, or this still fails --
# this is not a blanket exemption for that row, only for this exact,
# already-documented gap. Scoped per file: a mismatch in any other file,
# or any row not listed here, still fails --check.
CLASS_MEAN_ALLOWLIST = {
    "00_350_subject_work.md": [
        # SleepFM From-Scratch rows in the "Reported in the BNAIC paper"
        # tables (arXiv:2609.22105, Tables II-IV) -- values copied exactly
        # from the paper.
        ("SleepFM From-Scratch", 0.6582, 0.0060),  # EEG, 3 channels
        ("SleepFM From-Scratch", 0.3353, 0.0060),  # ECG only
        ("SleepFM From-Scratch", 0.6529, 0.0020),  # EEG + ECG
        # Pre-existing in the detail tables below (BIOT/From-Scratch
        # full-modality ablation, published-checkpoint 10-fold CV),
        # already flagged and left unaltered in an earlier task.
        ("EEG + EOG + ECG + respiration + EMG (all signals)", 0.6601, 0.0080),
        ("EEG + EOG", 0.6509, 0.0020),
        ("EEG + EOG + ECG + respiration", 0.6496, 0.0060),
        ("EEG + EOG + ECG", 0.6310, 0.0060),
        ("EEG + EOG + ECG + respiration + EMG (all signals)", 0.7305, 0.0013),
        ("EEG + EOG + ECG + respiration", 0.7269, 0.0034),
        ("EEG + EOG + ECG", 0.7237, 0.0060),
        ("EEG + EOG", 0.7205, 0.0071),
    ],
}


def class_mean_allowed(filename, row_label, macro, diff):
    for (label, allowed_macro, allowed_diff) in CLASS_MEAN_ALLOWLIST.get(filename, []):
        if label == row_label and abs(macro - allowed_macro) < 1e-6 and abs(diff - allowed_diff) < 5e-5:
            return True
    return False


def class_mean_check(table):
    """For tables with per-class columns Wake,N1,N2,N3,REM and a Macro F1
    column, check mean(per-class) == Macro F1 within 0.0006 per row.
    Skips a row whose exact (row label, macro F1, measured diff) is on
    CLASS_MEAN_ALLOWLIST for this file; any other mismatch still fails,
    including a changed diff for an otherwise-listed row."""
    issues = []
    needed = ["Macro F1", "Wake", "N1", "N2", "N3", "REM"]
    idx = {c: table.col_index(c) for c in needed}
    if any(v is None for v in idx.values()):
        return issues
    for ridx in table.data_rows():
        cells = split_row(table.lines[ridx])
        macro = cell_value(cells[idx["Macro F1"]])
        classes = [cell_value(cells[idx[c]]) for c in ["Wake", "N1", "N2", "N3", "REM"]]
        if macro is None or any(c is None for c in classes):
            continue
        mean_c = sum(classes) / 5
        diff = abs(mean_c - macro)
        if diff > 0.0006:
            row_label = cells[0].replace("**", "").strip()
            if class_mean_allowed(table.file, row_label, macro, diff):
                continue
            issues.append((table.file, ridx + 1, mean_c, macro))
    return issues


def numeric_tokens(text):
    return sorted(re.findall(r"-?\d+\.\d+", text))


def process_file(path, mode):
    """mode: 'check' or 'fix'. Returns (issues, lines_after)."""
    text = path.read_text()
    lines = text.splitlines()
    tables = find_tables(lines)
    for t in tables:
        t.file = path.name

    issues = []

    for t in tables:
        if t.spec is None:
            issues.append(f"{path.name}: table at line {t.header_line_idx + 1} ('{' | '.join(t.header)}') has no spec comment")
            continue
        if t.caption_line_idx is None:
            issues.append(f"{path.name}: table at line {t.header_line_idx + 1} missing caption line")

        # leftover-shorthand scan within this table's own text span
        span_start = t.spec_line_idx if t.spec_line_idx is not None else t.header_line_idx
        span_text = "\n".join(lines[span_start:t.rows_end])
        for pat in SHORTHAND_PATTERNS:
            if re.search(pat, span_text):
                issues.append(f"{path.name}: leftover shorthand '{pat}' near line {t.header_line_idx + 1}")

        # class-mean check
        for (fn, lineno, mean_c, macro) in class_mean_check(t):
            issues.append(f"{fn}:{lineno}: mean(per-class)={mean_c:.4f} != macro_f1={macro:.4f} (diff {abs(mean_c-macro):.4f})")

        # mark check (only meaningful in check mode, compare to what's there)
        if t.spec.get("marks") == "none":
            for ridx in t.data_rows():
                cell_text = lines[ridx]
                if "**" in cell_text or GREEN in cell_text or BLUE in cell_text:
                    issues.append(f"{path.name}:{ridx+1}: table marked marks=none but has a bold/square mark")
            continue

        value_cols = [c.strip() for c in t.spec.get("value_cols", "").split(",") if c.strip()]
        value_col_idx = [t.col_index(c) for c in value_cols]
        if any(idx is None for idx in value_col_idx):
            issues.append(f"{path.name}: table at line {t.header_line_idx+1} has value_cols not in header: {value_cols}")
            continue
        wanted = compute_marks(t)
        exclude_spec = t.spec.get("exclude_rows", "")
        for ridx in t.data_rows():
            cells = split_row(t.lines[ridx])
            excluded = row_excluded(cells, t.header, exclude_spec)
            for cidx, cell in enumerate(cells):
                want = wanted.get(ridx, {}).get(cidx, set()) if (cidx in value_col_idx and not excluded) else set()
                has_b = has_bold(cell)
                has_g = has_green(cell)
                has_bl = has_blue(cell)
                if has_bl:
                    issues.append(f"{path.name}:{ridx+1}: cell '{cell.strip()}' uses the legacy blue-square marker")
                if ("bold" in want) != has_b:
                    issues.append(f"{path.name}:{ridx+1} col '{t.header[cidx]}': bold should be {'on' if 'bold' in want else 'off'}, cell='{cell.strip()}'")
                if ("green" in want) != has_g:
                    issues.append(f"{path.name}:{ridx+1} col '{t.header[cidx]}': green should be {'on' if 'green' in want else 'off'}, cell='{cell.strip()}'")
                if cidx not in value_col_idx or excluded:
                    if has_b or has_g:
                        issues.append(f"{path.name}:{ridx+1} col '{t.header[cidx]}': non-value/excluded cell has a mark: '{cell.strip()}'")

    if mode == "fix":
        before_tokens = numeric_tokens(text)
        for t in tables:
            if t.spec is None:
                continue
            new_caption = make_caption(t.spec)
            new_spec_text = f"<!-- {spec_to_text(t.spec)} -->"
            if t.spec_line_idx is not None:
                lines[t.spec_line_idx] = new_spec_text
            if t.caption_line_idx is not None:
                lines[t.caption_line_idx] = new_caption
            rewrite_table_marks(t)
        new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        after_tokens = numeric_tokens(new_text)
        if before_tokens != after_tokens:
            raise RuntimeError(
                f"{path.name}: --fix would change numeric content! "
                f"removed={sorted(set(before_tokens) - set(after_tokens))} "
                f"added={sorted(set(after_tokens) - set(before_tokens))}"
            )
        return issues, new_text
    return issues, text


SHORTHAND_PATTERNS = [
    r"\bArm [A-E]\b", r"\bold350enc\b", r"\bseed123\b", r"\bch3\b",
    r"\bEEG_ONLY\b", r"\bEEG_ECG\b", r"\bECG_ONLY\b", r"\bEEG3_ONLY\b",
    # FS/NT are only flagged bare -- "SleepFM-FS" is the paper's own name for
    # SleepFM From-Scratch, legitimately cited as such in a Names-used table.
    r"(?<!-)\bFS\b", r"(?<!-)\bNT\b", r"\bZS\b", r"(?<![a-zA-Z0-9_])p10(?![a-zA-Z0-9_])",
    r"(?<![a-zA-Z0-9_])p20(?![a-zA-Z0-9_])", r"(?<![a-zA-Z0-9_])p50(?![a-zA-Z0-9_])",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()
    if not args.check and not args.fix:
        args.check = True

    files = sorted(RESULTS_DIR.glob("*.md"))
    all_issues = []
    for path in files:
        try:
            issues, new_text = process_file(path, "fix" if args.fix else "check")
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        all_issues.extend(issues)
        if args.fix:
            path.write_text(new_text)

    if args.check:
        print(f"Issues found: {len(all_issues)}")
        for i in all_issues:
            print(" -", i)
        sys.exit(1 if all_issues else 0)
    else:
        print(f"Fixed. ({len(all_issues)} issues were present in the spec/caption scan during this run; "
              f"marks and captions were rewritten from each table's spec regardless.)")
        sys.exit(0)


if __name__ == "__main__":
    main()
