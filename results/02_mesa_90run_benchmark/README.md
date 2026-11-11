# 02: MESA 90-run benchmark (current, authoritative)

**Status: current and authoritative.** This is the main MESA
full-cohort (2056-subject, `fold5_v1`, 5-fold CV) benchmark. Do not
modify any number in this family — only file *locations* were changed
in this reorganization (moved, not rewritten).

**One-line finding**: From-Scratch/Spectral/Next-Token (SleepFM
architecture, cohort-specific pretraining) outperform BIOT/LaBraM/
SensorLM (external-checkpoint-based) on 3-channel EEG and EEG+ECG;
ECG-only is the hardest condition for every model, with the largest
spread between models there.

## What's here

- `ALL_RESULTS_FULL.md` — the benchmark's own index/summary.
- `{FS,BIOT,LaBraM,NextToken,SensorLM,Spectral}_fullcohort_ALL_results.txt`
  — per-model, per-condition, per-fold macro F1/accuracy/per-class F1
  text tables (2056 subjects, `fold5_v1`).
- `per_subject/` — per-subject CSVs (subject-level macro F1/accuracy)
  for the models/conditions where that was captured.
- `epoch_predictions/` — raw per-epoch prediction `.npz` arrays for a
  subset of models/conditions.
- `per_run_metrics/` — `metrics.json` only for: the 9 plain
  From-Scratch/Next-Token/Spectral runs (`2026-09-17_2326`/
  `2026-09-19_1657`, 3 conditions × 3 pretrain methods) plus 3
  `seed123` robustness-check runs (From-Scratch, all 3 conditions,
  `2026-10-20`) — included here as a benchmark-robustness addendum,
  not a separate family, since no family was defined for seed checks.

## Regeneration

`sleepfm/pipeline/finetune_sleep_staging.py` (SleepFM variants) or the
model-specific script, against `sleepfm/configs/dataset_split_fold5_v1.json`,
patience 50 (SleepFM) or 20 (BIOT) per the model's own convention. ~10-30
min/fold depending on model; encoders themselves (`checkpoints/encoders/`)
must exist first — see the top-level `RESULTS.md` for where those live.
