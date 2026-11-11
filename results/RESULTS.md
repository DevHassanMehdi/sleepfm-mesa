# SleepFM-MESA: Results index

**Last updated**: 2026-11-11.

One line per family, the key table, and a link. Moody gets a link
only — no Moody numbers live in this repo (BDSP-licensed, owner-only
on scratch, never copied here).

## [00: 350-subject early work](00_350_subject_early_work/README.md) — superseded

Superseded by `02_mesa_90run_benchmark/`. Kept for history only.

## [01: Pipeline validation](01_pipeline_validation/README.md) — current

The full-cohort-vs-350-subject F1 gap was a label-corruption bug, not
a methodology problem. Fixed and validated (published-checkpoint
re-run, both pre- and post-fix generations; BIOT/LaBraM/SensorLM
re-run on the corrected cohort; zero metric mismatches against
independent recomputation).

## [02: MESA 90-run benchmark](02_mesa_90run_benchmark/README.md) — current, authoritative

| Model | EEG_ONLY | EEG_ECG | ECG_ONLY |
|---|---|---|---|
| From-Scratch | 0.7451±0.0052 | 0.7435±0.0035 | 0.5575±0.0058 |
| Next-Token | 0.7394±0.0051 | 0.7342±0.0039 | 0.4158±0.0074 |
| BIOT | 0.7541±0.0047 | 0.7479±0.0044 | 0.3235±0.0101 |

(Patience 50 for From-Scratch/Next-Token, 20 for BIOT — see
`02_mesa_90run_benchmark/README.md` for per-class detail.)

## [03: ECG ablation](03_ecg_ablation/README.md) — current

Pretraining-data ablation for the encoder; see
`03_ecg_ablation/ECG_ABLATION_RESULTS.md` for the conclusion.

## [04: EEG channel comparison](04_eeg_channel_comparison/README.md) — current

| Model | EEG1 | EEG2 | EEG3 | 3ch reference (p10) |
|---|---|---|---|---|
| From-Scratch | 0.6835±0.0120 | 0.6843±0.0100 | 0.6886±0.0121 | 0.7306±0.0102 |
| Next-Token | 0.7015±0.0063 | 0.6966±0.0098 | 0.6964±0.0091 | 0.7346±0.0053 |
| BIOT | 0.6976±0.0052 | 0.6887±0.0088 | 0.6998±0.0088 | 0.7541±0.0047 |

Single channel scores ~3-5 points below the matched 3-channel
reference for every model; which single channel barely matters.

## [05: Patience reruns (MESA only)](05_patience_reruns/README.md) — current

MESA beats Moody for every condition, every model, at matched
patience 20 — see `05_patience_reruns/patience_comparison_table.md`
for the full MESA-vs-Moody diff.

## [06: Moody / PhysioNet 2026 (I0006)](06_moody/README.md) — link only

Owner-only, outside git, on scratch. See the README for exactly which
notes files exist and what they cover.

## Where things live (data, encoders, checkpoints)

| What | Where |
|---|---|
| Raw MESA/MrOS/SHHS EDFs | `/scratch/project_2019517/sleepfm-data/{mesa,mros,shhs}/edf/` |
| Converted HDF5 | `sleepfm-data/{mesa,mros,shhs}/hdf5*` (see `../..` inventory notes for the `hdf5` vs `hdf5_350_puhti_baseline` duplicate finding) |
| Labels | `sleepfm-data/mesa/labels/` (current); `labels_CORRUPTED_pre_2026-08-06/` (deliberately retained reference, see `01_pipeline_validation/`) |
| Encoders (`best.pt`+`config.json`) | `sleepfm-data/checkpoints/encoders/` |
| Fine-tuned classifier checkpoints | `sleepfm-data/checkpoints/full_cohort/` |
| Embeddings (inputs to fine-tuning, not outputs) | Per-encoder, under `checkpoints/encoders/<encoder>/<arch>/<dataset>/`; the one legacy exception is `sleepfm-data/embeddings{,_5min_agg}/`, symlinked from `sleepfm/checkpoints/model_base/`, used by the published-checkpoint pipeline-validation family — see each family's own README for the exact regeneration command and cost |
| Moody data/encoders/checkpoints | `/scratch/project_2019517/moody2026_pilot/` — owner-only, never copied here |
| SLURM templates | `scripts/slurm_templates/` (copied into git this reorganization — see its own README) |
| Environments | `scripts/env/` (requirements + rebuild commands, this reorganization) |

## How to add a new experiment

1. Pick the next family number (or create a new one if it's genuinely
   a new experiment family, not a variant of an existing one).
2. `mkdir results/NN_short_name/`, write a `README.md` with a one-line
   finding, what's in the folder, and a regeneration recipe.
3. Copy only the small, aggregate files into git: `metrics.json` (never
   `per_subject_results.csv` or anything with per-subject/per-record
   IDs), text/markdown summaries. Raw pickles, full checkpoints, and
   per-subject CSVs stay on scratch (`experiments_full_cohort/`,
   gitignored) — link to them, don't copy them.
2. If the new experiment touches Moody data, it does **not** get a
   folder here — it goes in the owner-only Moody area, and this file
   gets a link-only entry like `06_moody`.
3. Add a one-line entry to this file's table of contents, in rank order
   by when the family was created (oldest first, matching the existing
   numbering).
4. Before committing: scan the diff for record IDs, tokens, and keys.
   Never commit a number that hasn't been independently recomputed
   from raw pickles at least once.
