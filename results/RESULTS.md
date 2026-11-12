# SleepFM-MESA: Results Index

**Last updated:** 2026-11-12.

## Files

*No marks: an index, not a ranked comparison.*
<!-- spec: items=files; marks=none; note=an index, not a ranked comparison -->
| File | Finding |
|---|---|
| [00_350_subject_work.md](00_350_subject_work.md) | BIOT scores best on EEG, 3 channels and EEG + ECG in the 350-subject work, superseded by 02. |
| [01_pipeline_validation.md](01_pipeline_validation.md) | A label-expansion bug capped every full-cohort score near 0.53, fixing it restored 0.70-0.78. |
| [02_mesa_benchmark.md](02_mesa_benchmark.md) | BIOT leads EEG-based conditions, SleepFM From-Scratch leads ECG-only by a wide margin. |
| [03_ecg_ablation.md](03_ecg_ablation.md) | SleepFM From-Scratch's ECG-only score rose from 0.39 to 0.56 mostly by fixing the training schedule. |
| [04_eeg_channels.md](04_eeg_channels.md) | No single EEG channel is clearly best for SleepFM From-Scratch or SleepFM Next-Token, BIOT's EEG, 1 channel (Cz-Oz) is clearly weakest. |
| [05_patience_reruns.md](05_patience_reruns.md) | Higher early-stopping patience helps a little and does not change which model wins. |
| [06_moody.md](06_moody.md) | Moody tracks MESA closely, MESA beats Moody on every condition for every model at matched early-stopping patience. |

See also [PIPELINE_VALIDATION_FINDINGS.md](../PIPELINE_VALIDATION_FINDINGS.md) for the full investigation log behind 01.

## Headline: best macro F1 per condition

*Compared: conditions, down each column, higher is better. Green square: best dataset setting along each row.*
<!-- spec: block=none; items=conditions; value_cols=MESA best,Moody best; bold=down; green=along -->
| Condition | MESA best | Moody best |
|---|---|---|
| EEG, 3 channels | **0.7541 (BIOT, early-stopping patience 20)** 🟩 | **0.7143 (SleepFM From-Scratch, retrained, early-stopping patience 50)** |
| EEG + ECG | 0.7479 (BIOT, early-stopping patience 20) 🟩 | **0.7145 (SleepFM From-Scratch, retrained, early-stopping patience 50)** |
| ECG only | 0.5575 (SleepFM From-Scratch, early-stopping patience 50) 🟩 | 0.5189 (SleepFM From-Scratch, retrained, early-stopping patience 50) |

## Where things live

*No marks: file-location reference.*
<!-- spec: items=locations; marks=none; note=file-location reference -->
| What | Where |
|---|---|
| Raw MESA EDFs | `/scratch/project_2019517/sleepfm-data/mesa/edf/` |
| Converted HDF5 | `mesa/hdf5` (live data path) and `mesa/hdf5_350_puhti_baseline` (350-subject baseline, byte-identical subset) |
| Labels | `sleepfm-data/mesa/labels/` (current), `labels_CORRUPTED_pre_2026-08-06/` (retained before-reference, see 01) |
| Encoders (`best.pt` + `config.json`) | `sleepfm-data/checkpoints/encoders/` |
| Fine-tuned classifier checkpoints | `sleepfm-data/checkpoints/full_cohort/` |
| Embeddings | Per-encoder, under `checkpoints/encoders/<encoder>/<arch>/<dataset>/` |
| Moody data, encoders, checkpoints | `/scratch/project_2019517/moody2026_pilot/`, owner-only. Moody's numbers are copied into 06, the records themselves stay there |
| SLURM templates | `scripts/slurm_templates/` |
| Environments | `scripts/env/` |
| Raw per-subject archive (pre-deletion snapshot) | `/projappl/project_2019517/archive/results_raw_2026-11-11.tar.gz` |

## Table standard

Every table in every `results/*.md` file is marked the same way, and checked
by `scripts/results_tables.py`.

**The marking rule:**

- Bold = the best value among the items being compared (models, pretraining
  setups, or channels), taken down each column within one block. A block is
  one condition. The first-column label is carried down over blank cells.
  Every value within 0.002 of the best is also bold.
- Green square (🟩) = the highest value along each row across the dataset
  columns (MESA, Moody zero-shot, Moody retrained). It appears only in
  tables that compare datasets. Ties within 0.002 all get it.
- Higher is better for macro F1, accuracy, and per-class F1.
- Drop, std, count, and patience columns never get a mark. Reference rows
  (for example "EEG, 3 channels" inside a single-channel comparison) are
  left out of the ranking and get no mark. Per-fold tables get no marks.
- Bold wraps the whole cell (value and ± std) and the square follows it:
  `**0.754**` 🟩, or `**0.7131 ± 0.0088**`.

**Layout:** Condition, then Model; numbers to 3 decimals in a summary table
or 4 decimals in a detail table, with ± std where the source has one;
early-stopping patience in its own column, never inside a model or row
name; descriptive names only (no internal shorthand like `EEG_ONLY`,
`p20`, or `Arm A` -- see each file's own Names-used table).

**The spec comment:** every table has an HTML comment directly above it
that drives its marks, and a plain-text italic caption (generated from the
spec) directly under the table's title:

```
### Heading

*Compared: models, down each column, higher is better. Green square: best
dataset setting along each row.*
<!-- spec: block=Condition; items=models; value_cols=MESA,Moody zero-shot,
Moody retrained; bold=down; green=along; exclude_rows=Model~EEG, 3 channels;
tie=0.002 -->
| table |
```

Spec fields: `block` (the column that groups rows into blocks, or `none`),
`items` (free text for the caption), `value_cols` (comma-separated column
names to rank), `bold` (`down` or `none`), `green` (`along` or `none`),
`exclude_rows` (`Column~substring`, rows to leave out of the ranking),
`tie` (tolerance, default 0.002), and `marks=none` for a table that gets no
marks at all (per-fold detail, names-used tables, reference tables).

**To add or change a table:** copy an existing table's spec comment, edit
it to match the new table's columns and comparison, then run

```
python scripts/results_tables.py --fix
python scripts/results_tables.py --check
```

`--fix` recomputes every caption and mark from the spec comments already in
the file and leaves every number untouched (it refuses to run if it would
change any number). `--check` reports anything still wrong: a missing or
incorrect mark, a missing spec or caption, leftover shorthand, or a
per-class row whose five class values don't average to its macro F1 within
0.0006.

## How to add a new experiment

1. Run it, land the aggregate `metrics.json` under `experiments_full_cohort/` on scratch, never commit per-subject files.
2. Add its macro F1 and per-class numbers to the matching flat file above, or start a new one if it's a new family.
3. Keep the file's own style: a title, a one-line Finding, a few short bullets, a Names-used table, the main table, then detail tables.
4. Use plain descriptive names in every header and row label; give early-stopping patience its own column, never bury it in a model or row name.
5. Add or copy a table spec per the "Table standard" above, then run `python scripts/results_tables.py --fix` and `--check`.
6. Before committing, scan the diff for record IDs, tokens, and keys.
