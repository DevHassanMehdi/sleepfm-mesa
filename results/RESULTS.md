# SleepFM-MESA: Results Index

**Last updated:** 2026-11-12.

## Files

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

| Condition | MESA best | Moody best |
|---|---|---|
| EEG, 3 channels | **0.7541** (BIOT, early-stopping patience 20) | 0.7143 (SleepFM From-Scratch, retrained, early-stopping patience 50) |
| EEG + ECG | 0.7479 (BIOT, early-stopping patience 20) | 0.7145 (SleepFM From-Scratch, retrained, early-stopping patience 50) |
| ECG only | 0.5575 (SleepFM From-Scratch, early-stopping patience 50) | 0.5189 (SleepFM From-Scratch, retrained, early-stopping patience 50) |

## Where things live

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

## How to add a new experiment

1. Run it, land the aggregate `metrics.json` under `experiments_full_cohort/` on scratch, never commit per-subject files.
2. Add its macro F1 and per-class numbers to the matching flat file above, or start a new one if it's a new family.
3. Keep the file's own style: a title, a one-line Finding, a few short bullets, a Names-used table, the main table, then detail tables.
4. Use plain descriptive names in every header and row label; give early-stopping patience its own column, never bury it in a model or row name.
5. Before committing, scan the diff for record IDs, tokens, and keys.
