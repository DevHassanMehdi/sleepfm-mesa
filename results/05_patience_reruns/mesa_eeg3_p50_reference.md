# MESA EEG3-only, patience 50 (reference run)

**Why this exists**: the EEG channel comparison (`../04_eeg_channel_comparison/`)
only has an EEG3-only number at patience 10. The Moody retraining task
needed a patience-50 MESA EEG3-only number to fairly compare against
Moody's own patience-50 EEG3-only result (which otherwise looked like
it "exceeded" MESA, an artifact of the patience-10 comparison). This
run isolates the patience effect from the channel effect.

Reuses the *existing* channel-comparison EEG3-only embeddings
(untouched, not regenerated). `pretrain_method=fromscratch_ch3_p50ref`
keeps the run distinct from both the 90-run benchmark (no EEG3_ONLY
condition there) and the patience-10 channel-comparison run.

| | Macro F1 (mean ± std, 5 folds) | Per-fold |
|---|---|---|
| Patience 10 (channel comparison, `../04_eeg_channel_comparison/`) | 0.6886 ± 0.0121 | — |
| **Patience 50 (this reference)** | **0.7100 ± 0.0021** | 0.7103, 0.7137, 0.7087, 0.7102, 0.7072 |

Recomputed independently from the raw test pickles (not
`compute_metrics.py`'s own printed output): exact match. Fold
disjointness confirmed: 411/411/411/411/412, zero overlap, union 2056.

Run name: `sleepfm__EEG3_ONLY__fromscratch_ch3_p50ref__fold5_v1__2026-11-09_212333`.

**Used by**: the Moody retraining notes
(`/scratch/project_2019517/moody2026_pilot/RETRAIN_PHASE1_RESULTS.md`,
owner-only) link here for this value rather than embedding it.

## Regeneration

`finetune_sleep_staging.py --config_path sleepfm/configs/config_ft_fromscratch_eegonly_fullcohort_EEG3_ONLY.yaml --patience 50 --pretrain_approach fromscratch_ch3_p50ref`,
5 folds, ~20-30 min/fold (reuses existing embeddings, no regeneration needed).
