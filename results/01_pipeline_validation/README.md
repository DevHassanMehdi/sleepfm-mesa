# 01: Pipeline validation (label bug + fold scheme + full-cohort integrity)

**One-line finding**: the full-cohort-vs-350-subject macro F1 gap seen
early in the project was caused by a label-corruption bug (not a
methodology or leakage problem); fixed, and the fix was validated by
re-running the published SleepFM checkpoint on both generations of the
full cohort (pre-fix 2026-08-06 and post-fix-plus-112-new-subjects
2026-08-08) and by re-running BIOT/LaBraM/SensorLM on the corrected
cohort, with zero metric mismatches against independent recomputation.

## What's here

- `PIPELINE_VALIDATION_FINDINGS.md` — the full investigation log (label
  bug discovery/root-cause/fix, full-cohort data integrity, folder
  structure redesign, the fold5_v1-vs-fold10_v1 decision, the F1-gap
  investigation across 8 sub-hypotheses, the window-size investigation,
  the channel-scope gap, MrOS/SHHS download bugs, and the final
  validation). This is the primary document — read it for the full
  story; this README is only a pointer.
- `LABEL_CHECK_350.md` — a focused, read-only re-verification of the
  "348 of 350" label-integrity figure against the pre-fix labels and
  the exact 350-subject manifest.
- `CV_COMPARISON_fold5_vs_fold10.md` — the fold5_v1 vs fold10_v1
  decision, moved here from `experiments_full_cohort/` (which is
  gitignored in its entirety; this one file was pulled out and
  tracked because it documents a standing methodology decision used
  by every family after it).
- `per_run_metrics/` — `metrics.json` only (no per-subject data) for
  every run referenced in the findings doc:
  - The published-checkpoint runs, **both generations**: `2026-08-06_1238`
    (pre-fix) and `2026-08-08_1533` (post-fix, 112 new subjects added,
    2056/2056), across all 4 published modality configs (`BAS`,
    `BAS_EKG`, `BAS_EKG_RESP`, `BAS_EKG_RESP_EMG` — note `BAS_EKG_RESP_EMG`
    only has the post-fix generation, no pre-fix run found).
  - The "Leg 1 re-run on the corrected cohort" (Section 9.5): BIOT,
    LaBraM, SensorLM, each on EEG_ONLY/EEG_ECG/ECG_ONLY,
    `2026-08-09_1229`/`2026-08-10_1230`.

## Regeneration

Re-run via `finetune_sleep_staging.py --pretrain_approach published`
(for the published checkpoint) or the model-specific fine-tune script,
against `sleepfm/configs/dataset_split_fold5_v1.json` and the current
(post-fix) `sleepfm-data/mesa/labels/`. Each run: ~10-20 min/fold on
the full 2056-subject cohort.
