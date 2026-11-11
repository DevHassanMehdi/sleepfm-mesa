# 04: EEG channel comparison (single-channel vs. 3-channel, patience 10)

**One-line finding**: each EEG channel on its own (EEG1/EEG2/EEG3)
scores ~0.68-0.70 across From-Scratch/Next-Token/BIOT, about 3-5 points
below the matched 3-channel reference — channel choice barely matters
on its own, but more channels together still help. See
`EEG_CHANNEL_RESULTS.md` for the full table and discussion.

## What's here

- `EEG_CHANNEL_RESULTS.md`, `EEG_CHANNEL_DETAILED.md` — summary and detail.
- `per_run_metrics/` — `metrics.json` only for:
  - 11 BIOT single-channel runs (EEG1/EEG2/EEG3_ONLY, patience 10 —
    scattered across multiple timestamped directories per channel,
    1-2 folds each, summing to 5 folds/channel; all copied).
  - 6 From-Scratch/Next-Token single-channel runs (EEG1/EEG2/EEG3_ONLY,
    `2026-10-27`).
  - 2 patience-10 **3-channel reference** reruns
    (`fromscratch_p10ref`/`nexttoken_p10ref`, `2026-10-28_154103`) —
    these exist specifically so the single-channel numbers have a
    patience-matched 3-channel comparison point (the main benchmark's
    own 3-channel number in `../02_mesa_90run_benchmark/` is patience
    50, not comparable at face value).

## Regeneration

Same `finetune_sleep_staging.py`/model-specific script, `--patience 10`
explicitly, `channel_like` set to the single channel (or the full
3-channel set for the `_p10ref` reruns). Each run ~5-15 min/fold
(single-channel runs are faster — less data per window).
