import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import results_tables as rt  # noqa: E402


def _process(text, mode="fix"):
    lines = text.splitlines()
    tables = rt.find_tables(lines)
    for t in tables:
        t.file = "synthetic.md"
    if mode == "fix":
        for t in tables:
            if t.spec is None:
                continue
            new_caption = rt.make_caption(t.spec)
            new_spec_text = f"<!-- {rt.spec_to_text(t.spec)} -->"
            if t.spec_line_idx is not None:
                lines[t.spec_line_idx] = new_spec_text
            if t.caption_line_idx is not None:
                lines[t.caption_line_idx] = new_caption
            rt.rewrite_table_marks(t)
        return "\n".join(lines)
    return lines, tables


def test_bold_down_column_basic():
    text = (
        "### Section\n\n"
        "*caption*\n"
        "<!-- spec: block=none; items=models; value_cols=Macro F1; bold=down; green=none -->\n"
        "| Model | Macro F1 |\n"
        "|---|---|\n"
        "| A | 0.700 |\n"
        "| B | 0.900 |\n"
        "| C | 0.899 |\n"
    )
    out = _process(text)
    assert "| A | 0.700 |" in out
    assert "| B | **0.900** |" in out
    # within 0.002 of 0.900 -> also bold
    assert "| C | **0.899** |" in out


def test_block_label_carried_over_blank_cells():
    text = (
        "*caption*\n"
        "<!-- spec: block=Condition; items=models; value_cols=Score; bold=down; green=none -->\n"
        "| Condition | Model | Score |\n"
        "|---|---|---|\n"
        "| X | A | 0.5 |\n"
        "|   | B | 0.9 |\n"
        "| Y | A | 0.3 |\n"
        "|   | B | 0.1 |\n"
    )
    out = _process(text)
    lines = out.splitlines()
    assert "**0.9**" in lines[-3]  # block X best
    assert "**0.3**" in lines[-2]  # block Y best (A, since 0.3 > 0.1)
    assert "**" not in lines[-1]


def test_ties_within_tolerance():
    text = (
        "*caption*\n"
        "<!-- spec: block=none; items=models; value_cols=Score; bold=down; green=none; tie=0.002 -->\n"
        "| Model | Score |\n"
        "|---|---|\n"
        "| A | 0.500 |\n"
        "| B | 0.501 |\n"
        "| C | 0.490 |\n"
    )
    out = _process(text)
    assert "**0.500**" in out
    assert "**0.501**" in out
    assert "0.490 |" in out and "**0.490**" not in out


def test_excluded_rows_never_marked_and_not_competition():
    text = (
        "*caption*\n"
        "<!-- spec: block=none; items=channels; value_cols=Score; bold=down; green=none; "
        "exclude_rows=Channel~All 3 -->\n"
        "| Channel | Score |\n"
        "|---|---|\n"
        "| Ch1 | 0.5 |\n"
        "| Ch2 | 0.6 |\n"
        "| All 3 channels | 0.9 |\n"
    )
    out = _process(text)
    lines = out.splitlines()
    assert "**0.6**" in lines[-2]
    assert "0.9" in lines[-1] and "**" not in lines[-1] and rt.GREEN not in lines[-1]


def test_green_along_row_with_tie():
    text = (
        "*caption*\n"
        "<!-- spec: block=none; items=models; value_cols=MESA,Moody; bold=none; green=along -->\n"
        "| Model | MESA | Moody |\n"
        "|---|---|---|\n"
        "| A | 0.700 | 0.500 |\n"
        "| B | 0.600 | 0.601 |\n"
    )
    out = _process(text)
    lines = out.splitlines()
    assert rt.GREEN in lines[-2] and "0.700" in lines[-2] and rt.GREEN not in lines[-2].split("|")[-2]
    # tie within 0.002 -> both get green
    assert lines[-1].count(rt.GREEN) == 2


def test_drop_column_never_marked():
    text = (
        "*caption*\n"
        "<!-- spec: block=none; items=models; value_cols=MESA,Moody; bold=down; green=along -->\n"
        "| Model | MESA | Moody | Drop |\n"
        "|---|---|---|---|\n"
        "| A | **0.700** | 0.500 | **0.200** |\n"
    )
    out = _process(text)
    assert "**0.200**" not in out
    assert "0.200" in out


def test_std_inside_cell_parsed_as_first_number():
    text = (
        "*caption*\n"
        "<!-- spec: block=none; items=models; value_cols=Macro F1; bold=down; green=none -->\n"
        "| Model | Macro F1 |\n"
        "|---|---|\n"
        "| A | 0.713 ± 0.009 |\n"
        "| B | 0.600 ± 0.001 |\n"
    )
    out = _process(text)
    assert "**0.713 ± 0.009**" in out
    assert "0.600 ± 0.001" in out and "**0.600" not in out


def test_na_and_dash_skipped():
    text = (
        "*caption*\n"
        "<!-- spec: block=none; items=models; value_cols=Score; bold=down; green=none -->\n"
        "| Model | Score |\n"
        "|---|---|\n"
        "| A | n/a |\n"
        "| B | -- |\n"
        "| C | 0.5 |\n"
    )
    out = _process(text)
    assert "**0.5**" in out
    assert "n/a" in out and "--" in out


def test_marks_none_strips_existing_marks():
    text = (
        "*caption*\n"
        "<!-- spec: items=foo; marks=none; note=reference -->\n"
        "| A | B |\n"
        "|---|---|\n"
        "| **0.5** 🟩 | 0.6 |\n"
    )
    out = _process(text)
    assert "**" not in out
    assert rt.GREEN not in out
    assert "0.5" in out and "0.6" in out


def test_idempotence():
    text = (
        "*caption*\n"
        "<!-- spec: block=none; items=models; value_cols=MESA,Moody; bold=down; green=along -->\n"
        "| Model | MESA | Moody |\n"
        "|---|---|---|\n"
        "| A | 0.700 | 0.500 |\n"
        "| B | 0.600 | 0.800 |\n"
    )
    once = _process(text)
    twice = _process(once)
    assert once == twice


def test_check_detects_missing_spec():
    text = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    lines, tables = _process(text, mode="check")
    assert tables and tables[0].spec is None


def test_class_mean_check():
    text = (
        "*caption*\n"
        "<!-- spec: block=none; items=models; value_cols=Macro F1; bold=down; green=none -->\n"
        "| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| A | 0.600 | 0.700 | 0.5 | 0.5 | 0.5 | 0.5 | 2.0 |\n"
    )
    lines, tables = _process(text, mode="check")
    t = tables[0]
    t.file = "x.md"
    issues = rt.class_mean_check(t)
    assert len(issues) == 1


def test_per_fold_table_marks_none_no_marks_added():
    text = (
        "*caption*\n"
        "<!-- spec: marks=none; note=per-fold detail -->\n"
        "| Model | Fold 0 | Fold 1 |\n"
        "|---|---|---|\n"
        "| A | 0.5 | 0.9 |\n"
    )
    out = _process(text)
    assert "**" not in out
    assert rt.GREEN not in out
