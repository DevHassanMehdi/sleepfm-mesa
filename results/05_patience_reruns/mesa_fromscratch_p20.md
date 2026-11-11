# MESA From-Scratch, patience 20 (matched-patience rerun)

**Why this exists**: to compare against Moody's own From-Scratch
patience-20 batch at matched patience (the main benchmark in
`../02_mesa_90run_benchmark/` is patience 50, BIOT is patience 20 —
this closes the gap for From-Scratch specifically).
`pretrain_approach=fromscratch_p20` keeps it distinct from the
benchmark's own `fromscratch` runs. Written to the default shared
checkpoint tree (not Moody's `--root`) — this is MESA data.

| Condition | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM | Per-fold macro F1 |
|---|---|---|---|---|---|---|---|---|
| EEG_ONLY (3ch) | 0.7428±0.0045 | 0.8392 | 0.9467 | 0.4609 | 0.8165 | 0.6551 | 0.8348 | [0.7399, 0.7488, 0.7400, 0.7475, 0.7377] |
| EEG3_ONLY | 0.6998±0.0061 | 0.8060 | 0.9202 | 0.3699 | 0.7830 | 0.6428 | 0.7833 | [0.6939, 0.7115, 0.6964, 0.6977, 0.6996] |
| EEG_ECG | 0.7395±0.0044 | 0.8377 | 0.9459 | 0.4470 | 0.8144 | 0.6607 | 0.8297 | [0.7376, 0.7460, 0.7337, 0.7429, 0.7375] |
| ECG_ONLY | 0.5431±0.0126 | 0.7027 | 0.8717 | 0.1415 | 0.6505 | 0.4231 | 0.6288 | [0.5473, 0.5482, 0.5194, 0.5568, 0.5438] |

Recomputed from raw test pickles (exact match to printed values). Fold
disjointness: [411,411,411,411,412], all pairwise disjoint, union 2056,
for all 4 conditions. Per-class-mean-equals-macro-F1 sanity check:
passed exactly for every row.

20 runs (4 conditions × 5 folds), smoke-checked (EEG_ONLY fold 0) before
the other 19, all exit 0, no record-ID leaks. GPU-hours: 3.60.

## Regeneration

`finetune_sleep_staging.py --patience 20 --pretrain_approach fromscratch_p20`,
configs: `config_finetune_full_cohort{,_ECG_ONLY,_EEG_ECG}.yaml` +
`config_ft_fromscratch_eegonly_fullcohort_EEG3_ONLY.yaml`, split
`sleepfm/configs/dataset_split_fold5_v1.json`. ~5-15 min/fold.
