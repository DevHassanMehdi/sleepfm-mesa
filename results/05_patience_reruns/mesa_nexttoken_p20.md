# MESA Next-Token, patience 20 (matched-patience rerun)

**Why this exists**: same reasoning as `mesa_fromscratch_p20.md` —
matched-patience comparison against Moody's Next-Token patience-20
batch. `pretrain_approach=nexttoken_p20`, distinct from the benchmark's
own `nexttoken` runs. Written to the default shared checkpoint tree.

| Condition | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM | Per-fold macro F1 |
|---|---|---|---|---|---|---|---|---|
| EEG_ONLY (3ch) | 0.7350±0.0051 | 0.8359 | 0.9448 | 0.4300 | 0.8149 | 0.6684 | 0.8171 | [0.7381, 0.7399, 0.7353, 0.7367, 0.7252] |
| EEG3_ONLY | 0.7138±0.0073 | 0.8159 | 0.9256 | 0.3953 | 0.7925 | 0.6589 | 0.7969 | [0.7032, 0.7201, 0.7233, 0.7141, 0.7086] |
| EEG_ECG | 0.7322±0.0024 | 0.8322 | 0.9439 | 0.4295 | 0.8078 | 0.6631 | 0.8169 | [0.7337, 0.7359, 0.7311, 0.7289, 0.7315] |
| ECG_ONLY | 0.4190±0.0049 | 0.5980 | 0.8264 | 0.0106 | 0.5147 | 0.3120 | 0.4314 | [0.4150, 0.4239, 0.4158, 0.4260, 0.4144] |

Recomputed from raw test pickles (exact match). Fold disjointness:
[411,411,411,411,412], all pairwise disjoint, union 2056. Per-class-mean-
equals-macro-F1 sanity check: passed exactly for every row.

20 runs (4 conditions × 5 folds), smoke-checked (EEG_ONLY fold 0) first,
all exit 0, no record-ID leaks. GPU-hours: 3.68.

## Regeneration

`finetune_sleep_staging.py --patience 20 --pretrain_approach nexttoken_p20`,
configs: `config_ft_nexttoken_eegonly_fullcohort_EEG_ONLY.yaml`,
`config_finetune_full_cohort_{ECG_ONLY,EEG_ECG}_nexttoken.yaml`,
`config_ft_nexttoken_eegonly_fullcohort_EEG3_ONLY.yaml`. ~5-15 min/fold.
