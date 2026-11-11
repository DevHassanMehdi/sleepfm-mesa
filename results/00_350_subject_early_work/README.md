# 00: 350-subject early work (superseded)

**Status: superseded.** Kept for historical reference only — every number
here was later reproduced and extended by the 2056-subject full-cohort
work in `../02_mesa_90run_benchmark/`. Do not cite these for current
comparisons.

## What's here

Early-era text summaries from before the full-cohort (2056-subject)
pipeline existed, run on a 350-subject MESA subset:

- `FS_SleepFM_results.txt`, `FS_EEG_ECG_results.txt` — From-Scratch SleepFM
- `BIOT_ALL_results.txt`, `LaBraM_ALL_results.txt`, `SensorLM_ALL_results.txt`,
  `MOMENT_ALL_results.txt`, `Spectral_ALL_results.txt`, `NextToken_ALL_results.txt`,
  `Combined_ALL_results.txt` — other models, same 350-subject era
- `USleep_ALL_results.txt`, `YASA_ALL_results.txt` — non-learned baselines
- `PUB_SleepFM_results.txt` — published-SleepFM checkpoint, 350-subject era
  (precursor to the full validation in `../01_pipeline_validation/`)
- `ALL_RESULTS.md` — the era's own aggregator/index, covering the
  published encoder and the early BAS/Spectral/Combined/Next-Token
  modality ablation (placement is a judgment call — this file predates
  the full-cohort benchmark and uses the old BAS/EKG/EMG/RESP channel
  scheme throughout, so it reads as 350-subject-era, not current benchmark)

## No supporting files

None of the files in this family have accompanying raw pickles,
per-subject CSVs, or `metrics.json` anywhere found in this repo or on
scratch — they are text summaries only, and cannot be independently
re-verified anymore. This is expected and not a loss: every number was
superseded.

## Regeneration

Not applicable — superseded by `../02_mesa_90run_benchmark/`.
