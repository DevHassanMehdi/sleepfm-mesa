# EEG Channel Statistics and Compute Analysis (Phase 1)

**Finding:** Single EEG channels are mostly statistically indistinguishable from each other, but every single channel is a real, non-equivalent drop from its model's 3-channel reference (0.02-0.05 macro F1, TOST rejects equivalence at both 0.01 and 0.02 in every case). BIOT beats each SleepFM model on roughly half of their head-to-head comparisons, for a compute cost about 39x the FLOPs/epoch of the SleepFM encoder.

- Channel-pair comparisons (EEG1 vs EEG2 vs EEG3) are mostly TOST-equivalent at 0.01; where Friedman finds a difference across all three (every model, p < 0.0001), Holm-adjusted post-hoc pairs are small (median diff <= 0.008 macro F1) and mixed in sign across models.
- Single-channel-vs-reference comparisons are never equivalent, at either margin, for any model: all 9 such pairs (3 channels x 3 models) have a 90% CI entirely below -0.01 and below -0.02.
- Head-to-head, BIOT scores significantly higher macro F1 than SleepFM From-Scratch on EEG1, EEG3, and the 3-channel reference (not EEG2), and higher than SleepFM Next-Token only on the 3-channel reference (lower on EEG1/EEG2, no difference on EEG3) -- 4 of 8 head-to-head comparisons significantly favor BIOT, 2 favor the SleepFM model, 2 show no difference. BIOT's FLOPs/epoch is about 39x the SleepFM encoder's, at both 1 and 3 channels -- see Part E6.

## Names used

*No marks: names and descriptions.*
<!-- spec: items=names and descriptions; marks=none; note=names and descriptions -->
| Name in this file | Plain description |
|---|---|
| SleepFM From-Scratch | SleepFM's encoder, trained from random initialization with a contrastive pretraining objective, then fine-tuned for sleep staging |
| SleepFM Next-Token | SleepFM's encoder, pretrained with a next-token-prediction objective, then fine-tuned for sleep staging |
| BIOT | The published BIOT EEG foundation model, fine-tuned for sleep staging |
| EEG1 only | A single EEG lead, bipolar Fz(+)-Cz(-) |
| EEG2 only | A single EEG lead, bipolar Cz(+)-Oz(-) |
| EEG3 only | A single EEG lead, bipolar C4(+)-M1(-), the exact electrode match to MESA's own C4-M1 lead |
| 3-channel, patience 10 | All three EEG leads together, early-stopping patience 10 -- the matched-patience reference for the single-channel SleepFM runs |
| 3-channel benchmark | The main 3-channel run per model: patience 50 for SleepFM From-Scratch/Next-Token, patience 20 for BIOT (BIOT has no patience-10 3-channel run) |
| Subject | A MESA participant; the unit of resampling and testing throughout Part D, never an epoch or a fold |
| Cluster bootstrap | Resampling subjects (not epochs) with replacement, 2000 resamples, for a 95% CI of a pooled statistic |
| Rank-biserial | Effect size for the paired Wilcoxon signed-rank test, in [-1, 1]; sign follows condition A minus condition B |
| Holm adjustment | Step-down p-value correction across the pairs tested within one model/stage, controlling the family-wise error rate |
| TOST | Two one-sided test for equivalence: a result is "equivalent" at a margin when the 90% CI of the paired difference lies entirely inside the margin |
| FLOPs | Multiply-accumulate-dominated floating point operations, counted by torch.utils.flop_counter.FlopCounterMode -- see Part E2 for exactly what is and is not counted |
| Shared-node | A benchmark run on the normal shared GPU partition, alongside other users' jobs on the same physical node |
| Exclusive-node | A benchmark run with the whole node (all 4 GPUs, all CPUs) reserved, no other job running on it |

**Setup:** Full MESA cohort, 2056 subjects, fold5_v1. Scope: SleepFM From-Scratch, SleepFM Next-Token, BIOT; EEG1/EEG2/EEG3 single-channel patience-10 runs, the 3-channel patience-10 reference (SleepFM only), and the 3-channel benchmark run per model. Folds 0-4.

**Legend:** Bold = best model down each column within each condition (see each table's caption, and the "Table standard" in RESULTS.md). Drop, CI, p-value, and effect-size columns are never marked; D.2-D.7 and most of Part E are marks=none tables (comparison statistics, not head-to-head rankings).

## Summary of tests

*No marks: summary of what was run and what passed.*
<!-- spec: items=test summary; marks=none; note=summary of what was run and what passed -->
| Check | Result |
|---|---|
| Part A inventory (70 run-folders: 50 SleepFM + 20 BIOT) | complete, 0 missing fold, 0 missing directory |
| Part B: BIOT macro F1/accuracy match archived metrics (tolerance 0.0001) | 20/20 match, diff=0.0 on every run |
| Part B: no checkpoint-folder or experiments_full_cohort file changed/added/removed after the analysis rerun | confirmed (full before/after file listing compared) |
| Part C: every (model, condition) covers the same 2056 subjects | OK |
| Part C: mean of 5 per-class values equals macro F1 (tolerance 0.0006) | OK (built directly from sklearn f1_score per class; see note below D.1) |
| Part C: undefined-value counts | kappa=0, f1/recall_N1=294, f1/recall_N3=2394, f1/recall_REM=616 (table below) |
| Part D.1: pooled/fold-mean reproduce the documented 3-channel-benchmark fold means (02/04/06) | OK, diff=0.0000 on all 3 |
| Part E5: training cost from sacct | not recoverable -- see note below |
| `scripts/results_tables.py --check` | 0 issues (run after --fix, see Part F commit) |
| `scripts/check_results_numbers.py` | run after --fix; see commit |
| Unit tests (test_channel_stats.py, test_compute_profile.py, test_experiment_paths.py, full suite) | 65 passed |
| Link check | 0 broken links |

**E5 note:** training cost (GPU-hours, wall time, epochs to stop, best epoch) for the original pretraining and fine-tuning jobs in this task's scope could not be recovered. `sacct` has no records before approximately 2026-11-11 (checked explicitly: querying August 2026 returns nothing), and `logs/` in this repository holds only files from the current migration-era work -- nothing from when these runs were originally trained (July-October 2026). Checkpoint `config.json` files record `epochs` as the configured maximum (e.g. 500), not the actual stop epoch, and no `best_epoch` field is saved anywhere. This is a hard data-availability gap, not a measurement choice; no numbers are given in its place.

## Part A: inventory

*No marks: inventory, not a ranked comparison.*
<!-- spec: items=run-folder inventory; marks=none; note=inventory, not a ranked comparison -->
| Model | Condition | Folds | Run folder(s) | Raw predictions |
|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | 0-4 | sleepfm__EEG1_ONLY__fromscratch__fold5_v1__2026-10-27_1521 | 5/5 raw pickles, all 5 folds |
| SleepFM Next-Token | EEG1 only | 0-4 | sleepfm__EEG1_ONLY__nexttoken__fold5_v1__2026-10-27_1522 | 5/5 raw pickles, all 5 folds |
| SleepFM From-Scratch | EEG2 only | 0-4 | sleepfm__EEG2_ONLY__fromscratch__fold5_v1__2026-10-27_153632 | 5/5 raw pickles, all 5 folds |
| SleepFM Next-Token | EEG2 only | 0-4 | sleepfm__EEG2_ONLY__nexttoken__fold5_v1__2026-10-27_153632_y | 5/5 raw pickles, all 5 folds |
| SleepFM From-Scratch | EEG3 only | 0-4 | sleepfm__EEG3_ONLY__fromscratch__fold5_v1__2026-10-27_153632_x | 5/5 raw pickles, all 5 folds |
| SleepFM Next-Token | EEG3 only | 0-4 | sleepfm__EEG3_ONLY__nexttoken__fold5_v1__2026-10-27_153632_z | 5/5 raw pickles, all 5 folds |
| SleepFM From-Scratch | 3-channel, patience 10 | 0-4 | sleepfm__EEG_ONLY__fromscratch_p10ref__fold5_v1__2026-10-28_154103 | 5/5 raw pickles, all 5 folds |
| SleepFM Next-Token | 3-channel, patience 10 | 0-4 | sleepfm__EEG_ONLY__nexttoken_p10ref__fold5_v1__2026-10-28_154103_nt | 5/5 raw pickles, all 5 folds |
| SleepFM From-Scratch | 3-channel benchmark | 0-4 | sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-09-19_1657 | 5/5 raw pickles, all 5 folds |
| SleepFM Next-Token | 3-channel benchmark | 0-4 | sleepfm__EEG_ONLY__nexttoken__fold5_v1__2026-09-17_2326 | 5/5 raw pickles, all 5 folds |
| BIOT | EEG1 only | 0-4 | 4 run folders (one timestamp per fold-batch): 2026-10-27_1523(f0), 2026-10-28_1541(f1,f2), 2026-10-28_1543(f3), 2026-10-28_1544(f4) | before Part B: only best.pth/config.json/latest.pth live; after: raw predictions in `~/analysis/eeg_channels/biot_predictions/` |
| BIOT | EEG2 only | 0-4 | 4 run folders: 2026-10-28_1544(f0), 1547(f1), 1549(f2), 1551(f3,f4) | same as above |
| BIOT | EEG3 only | 0-4 | 3 run folders: 2026-10-28_1553(f0,f1), 1554(f2,f3), 1555(f4) | same as above |
| BIOT | 3-channel benchmark | 0-4 | biot__EEG_ONLY__finetuned__fold5_v1__2026-08-09_1229 | same as above |

70 run-folders total (50 SleepFM + 20 BIOT), 0 missing.

## Part B: BIOT raw-prediction rerun
`evaluate_biot.py` previously had no way to save per-window predictions (only the aggregate `metrics.json` and a per-subject `macro_f1`/`accuracy`/`n_valid_windows` CSV, no raw labels/predictions/probabilities). Added `--save_predictions` (default off): writes labels, predictions, softmax probabilities, and a parallel subject-id array to `~/analysis/eeg_channels/biot_predictions/<run_name>/fold_<N>/predictions.npz` -- never into a checkpoint folder. Reran all 20 BIOT runs in scope with it.

All 20 reruns matched their archived `metrics.json` exactly (diff=0.0 on both macro F1 and accuracy, well inside the 0.0001 tolerance) -- condensed by condition below; every individual fold matched just as exactly.

*No marks: verification log, not a ranked comparison.*
<!-- spec: items=verification summary; marks=none; note=verification log, not a ranked comparison -->
| Condition | Folds matching (tolerance 0.0001) | Macro F1 / accuracy diff |
|---|---|---|
| EEG1 only | 5/5 | 0.0 on every fold |
| EEG2 only | 5/5 | 0.0 on every fold |
| EEG3 only | 5/5 | 0.0 on every fold |
| 3-channel benchmark | 5/5 | 0.0 on every fold |

**Checkpoint-folder incident:** the first rerun attempt (before `--no_checkpoint_link` existed) wrote into all 20 checkpoint folders' `config.json` via `write_metrics_bundle`'s existing, unconditional call to `link_checkpoint_and_results` (updates `results_dir` to the current path) -- caught, reverted (diffed against the one file read in full during Part A, and against the deterministic old-account-path reconstruction for the other 19; confirmed only `results_dir` ever differed), and patched: `sleepfm/experiment_paths.py`'s `link_checkpoint_and_results` now takes `skip_checkpoint_side`, `write_metrics_bundle` takes `no_checkpoint_link`, and `evaluate_biot.py`/`evaluate_labram.py`/`evaluate_sensorlm.py` (which all share this function) each got a `--no_checkpoint_link` flag (default off, so training and any existing caller are unaffected). The verified rerun above used it for all 20 runs. `write_metrics_bundle` still writes its results bundle under `experiments_full_cohort/` (never a checkpoint folder, but also not `~/analysis/`) regardless of the flag -- each run's bundle was deleted immediately after its metrics were checked against the archive, confirmed by a full before/after file listing of both `data/checkpoints/full_cohort/` and `experiments_full_cohort/full_cohort/`: **no file under either tree was changed, added, or removed** by the verified rerun.

## Part C: per-subject table
Saved to `~/analysis/eeg_channels/per_subject_metrics.csv` (28,784 rows: 14 (model, condition) groups x 2056 subjects, not in git): one row per (model, condition, fold, subject_id), with `n_valid_epochs`, `accuracy`, `kappa`, `macro_f1`, and `f1_<Stage>`/`recall_<Stage>` for Wake/N1/N2/N3/REM, computed from the mask (valid epochs only) and over only the classes present in that subject. A second file, `fold_level_metrics.csv` (70 rows), holds the epoch-flattened per-fold macro F1/accuracy used by D.1 and the reproduction check above -- a different statistic from the per-subject table, see D.1's note.


*No marks: undefined-value counts, not a ranked comparison.*
<!-- spec: items=undefined-value counts; marks=none; note=undefined-value counts, not a ranked comparison -->
| Metric | Rows with an undefined value (of 28,784) |
|---|---|
| kappa | 0 |
| f1_Wake / recall_Wake | 0 |
| f1_N1 / recall_N1 | 294 |
| f1_N2 / recall_N2 | 0 |
| f1_N3 / recall_N3 | 2394 |
| f1_REM / recall_REM | 616 |

`kappa` is undefined when a subject's targets AND predictions are both constant (never happened here). `f1_<Stage>`/`recall_<Stage>` are undefined when that stage never occurs in a subject's valid epochs for that night. Each undefined-row count divides evenly by 14 (the number of (model, condition) groups): 294/14=21, 2394/14=171, 616/14=44 -- the same 21/171/44 subjects lack N1/N3/REM respectively in every condition, because this is a property of their one night's ground-truth labels, not of the model or channel used to predict it. This cross-check held exactly.

## Part D: statistics

### D.1: pooled macro F1 vs. the established fold mean

*Compared: models, down each column within each block, higher is better.*
<!-- spec: block=Condition; items=models; value_cols=Pooled macro F1 (95% CI),Fold mean +/- std (epoch-flattened); bold=down; green=none -->
| Condition | Model | Pooled macro F1 (95% CI) | Fold mean +/- std (epoch-flattened) | n subjects |
|---|---|---|---|---|
| 3-channel benchmark | BIOT | **0.7084 [0.7036, 0.7130]** | **0.7541 +/- 0.0047** | 2056 |
| EEG1 only | BIOT | 0.6498 [0.6443, 0.6554] | 0.6976 +/- 0.0053 | 2056 |
| EEG2 only | BIOT | 0.6415 [0.6356, 0.6473] | 0.6887 +/- 0.0088 | 2056 |
| EEG3 only | BIOT | 0.6538 [0.6479, 0.6596] | **0.6998 +/- 0.0088** | 2056 |
| 3-channel benchmark | SleepFM From-Scratch | 0.6984 [0.6938, 0.7031] | 0.7451 +/- 0.0052 | 2056 |
| 3-channel, patience 10 | SleepFM From-Scratch | 0.6862 [0.6817, 0.6909] | 0.7306 +/- 0.0102 | 2056 |
| EEG1 only | SleepFM From-Scratch | 0.6415 [0.6363, 0.6469] | 0.6835 +/- 0.0120 | 2056 |
| EEG2 only | SleepFM From-Scratch | 0.6427 [0.6371, 0.6486] | 0.6843 +/- 0.0100 | 2056 |
| EEG3 only | SleepFM From-Scratch | 0.6427 [0.6373, 0.6481] | 0.6886 +/- 0.0121 | 2056 |
| 3-channel benchmark | SleepFM Next-Token | 0.6960 [0.6914, 0.7009] | 0.7394 +/- 0.0051 | 2056 |
| 3-channel, patience 10 | SleepFM Next-Token | **0.6929 [0.6885, 0.6974]** | **0.7346 +/- 0.0053** | 2056 |
| EEG1 only | SleepFM Next-Token | **0.6625 [0.6568, 0.6680]** | **0.7014 +/- 0.0063** | 2056 |
| EEG2 only | SleepFM Next-Token | **0.6596 [0.6538, 0.6656]** | **0.6966 +/- 0.0098** | 2056 |
| EEG3 only | SleepFM Next-Token | **0.6584 [0.6528, 0.6642]** | 0.6964 +/- 0.0091 | 2056 |

**Why pooled and fold-mean differ (consistently, by about 0.04-0.05 here):** "Pooled macro F1" is the mean of each subject's OWN macro F1 (computed from only that subject's night, over only the classes present in it), bootstrapped over subjects -- exactly the statistic the task's "resample subjects" bootstrap specifies. "Fold mean" is the already-published statistic (results/02, 04, 06): pool every subject's valid epochs within a fold first, then compute ONE macro F1 over the fixed 5 classes for that fold. These differ because macro F1 is a nonlinear (harmonic-mean-based) function of per-class counts, and several classes -- especially N1, and N3/REM for some subjects -- occur only a handful of times in a single subject's single night. A few misclassified epochs in a class with few occurrences swings that subject's F1 for that class drastically; across 2056 subjects this does not average out the way linear statistics would, producing a systematic downward bias in the subject-averaged statistic relative to the epoch-pooled one. This is not a measurement error: the fold-mean column reproduces the documented values exactly (diff=0.0000 for all three 3-channel-benchmark rows, checked explicitly before accepting this table).

### D.2: paired differences within each model

*No marks: paired comparison statistics, not a ranked comparison.*
<!-- spec: items=paired comparisons; marks=none; note=paired comparison statistics, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | Median diff macro F1 | Median diff accuracy | Median diff kappa | Wilcoxon p | Rank-biserial | Bootstrap 95% CI (pooled macro F1 diff) | Holm p | n subjects |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | -0.0051 | 0.0009 | 0.0005 | 0.0001 | -0.099 | [-0.0052, 0.0031] | 0.0003 | 2056 |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 0.0022 | 0.0049 | 0.0067 | 0.4594 | 0.019 | [-0.0063, 0.0042] | 0.4594 | 2056 |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | 0.0058 | 0.0034 | 0.0057 | 0.0099 | 0.066 | [-0.0049, 0.0056] | 0.0198 | 2056 |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0333 | -0.0154 | -0.0240 | 5.19e-142 | -0.646 | [-0.0484, -0.0412] | 3.12e-141 | 2056 |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0283 | -0.0168 | -0.0258 | 3.65e-137 | -0.635 | [-0.0472, -0.0400] | 1.83e-136 | 2056 |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0319 | -0.0181 | -0.0283 | 8.62e-133 | -0.624 | [-0.0472, -0.0399] | 3.45e-132 | 2056 |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | -0.0029 | 0.0000 | -0.0010 | 0.0480 | -0.050 | [-0.0013, 0.0073] | 0.0480 | 2056 |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.0052 | 0.0008 | 0.0017 | 0.0037 | 0.074 | [-0.0014, 0.0097] | 0.0074 | 2056 |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 0.0076 | 0.0000 | 0.0019 | 1.38e-05 | 0.111 | [-0.0044, 0.0070] | 4.15e-05 | 2056 |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0194 | -0.0131 | -0.0191 | 6.35e-84 | -0.494 | [-0.0341, -0.0268] | 3.17e-83 | 2056 |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0177 | -0.0128 | -0.0180 | 9.05e-83 | -0.491 | [-0.0369, -0.0296] | 3.62e-82 | 2056 |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0229 | -0.0130 | -0.0188 | 2.92e-93 | -0.522 | [-0.0383, -0.0307] | 1.75e-92 | 2056 |
| BIOT | EEG1 only | EEG2 only |  | 0.0051 | 0.0000 | -0.0001 | 0.0010 | 0.084 | [0.0040, 0.0126] | 0.0020 | 2056 |
| BIOT | EEG1 only | EEG3 only |  | -0.0006 | 0.0000 | 0.0004 | 0.1587 | -0.036 | [-0.0096, 0.0014] | 0.1587 | 2056 |
| BIOT | EEG2 only | EEG3 only |  | -0.0071 | 0.0007 | -0.0001 | 3.97e-07 | -0.129 | [-0.0179, -0.0063] | 1.19e-06 | 2056 |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0423 | -0.0333 | -0.0466 | 1.04e-234 | -0.833 | [-0.0625, -0.0549] | 5.22e-234 | 2056 |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0489 | -0.0325 | -0.0468 | 7.50e-244 | -0.849 | [-0.0709, -0.0630] | 4.50e-243 | 2056 |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0395 | -0.0322 | -0.0450 | 3.08e-201 | -0.771 | [-0.0586, -0.0508] | 1.23e-200 | 2056 |

Diff = condition A minus condition B. Holm adjustment is applied within each model across its 6 pairs (EEG1-EEG2, EEG1-EEG3, EEG2-EEG3, and each of the three channels vs. that model's reference).

### D.3: TOST equivalence

*No marks: equivalence verdicts, not a ranked comparison.*
<!-- spec: items=TOST verdicts; marks=none; note=equivalence verdicts, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | Margin | Point diff | 90% CI (pooled macro F1 diff) | Verdict |
|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | 0.01 | -0.0012 | [-0.0047, 0.0024] | equivalent |
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | 0.02 | -0.0012 | [-0.0047, 0.0024] | equivalent |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 0.01 | -0.0012 | [-0.0055, 0.0032] | equivalent |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 0.02 | -0.0012 | [-0.0055, 0.0032] | equivalent |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | 0.01 | -0.0000 | [-0.0042, 0.0046] | equivalent |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | 0.02 | -0.0000 | [-0.0042, 0.0046] | equivalent |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0447 | [-0.0478, -0.0416] | not equivalent |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0447 | [-0.0478, -0.0416] | not equivalent |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0435 | [-0.0465, -0.0405] | not equivalent |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0435 | [-0.0465, -0.0405] | not equivalent |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0435 | [-0.0466, -0.0405] | not equivalent |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0435 | [-0.0466, -0.0405] | not equivalent |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 0.01 | 0.0029 | [-0.0008, 0.0067] | equivalent |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 0.02 | 0.0029 | [-0.0008, 0.0067] | equivalent |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.01 | 0.0041 | [-0.0004, 0.0087] | equivalent |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.02 | 0.0041 | [-0.0004, 0.0087] | equivalent |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 0.01 | 0.0011 | [-0.0034, 0.0061] | equivalent |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 0.02 | 0.0011 | [-0.0034, 0.0061] | equivalent |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0304 | [-0.0334, -0.0273] | not equivalent |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0304 | [-0.0334, -0.0273] | not equivalent |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0333 | [-0.0364, -0.0303] | not equivalent |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0333 | [-0.0364, -0.0303] | not equivalent |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0345 | [-0.0377, -0.0313] | not equivalent |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0345 | [-0.0377, -0.0313] | not equivalent |
| BIOT | EEG1 only | EEG2 only |  | 0.01 | 0.0083 | [0.0046, 0.0120] | inconclusive |
| BIOT | EEG1 only | EEG2 only |  | 0.02 | 0.0083 | [0.0046, 0.0120] | equivalent |
| BIOT | EEG1 only | EEG3 only |  | 0.01 | -0.0041 | [-0.0088, 0.0007] | equivalent |
| BIOT | EEG1 only | EEG3 only |  | 0.02 | -0.0041 | [-0.0088, 0.0007] | equivalent |
| BIOT | EEG2 only | EEG3 only |  | 0.01 | -0.0123 | [-0.0171, -0.0074] | inconclusive |
| BIOT | EEG2 only | EEG3 only |  | 0.02 | -0.0123 | [-0.0171, -0.0074] | equivalent |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.01 | -0.0586 | [-0.0617, -0.0555] | not equivalent |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.02 | -0.0586 | [-0.0617, -0.0555] | not equivalent |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.01 | -0.0669 | [-0.0702, -0.0635] | not equivalent |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.02 | -0.0669 | [-0.0702, -0.0635] | not equivalent |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.01 | -0.0546 | [-0.0581, -0.0514] | not equivalent |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.02 | -0.0546 | [-0.0581, -0.0514] | not equivalent |

Margins (0.01 and 0.02 macro F1) were fixed before looking at any result, per the task instructions. "Equivalent" requires the 90% CI entirely inside [-margin, +margin]; "not equivalent" requires it entirely outside on one side; otherwise "inconclusive".

### D.4: Friedman test across EEG1/EEG2/EEG3

*No marks: omnibus test, not a ranked comparison.*
<!-- spec: items=Friedman test results; marks=none; note=omnibus test, not a ranked comparison -->
| Model | n subjects | Friedman statistic | p-value |
|---|---|---|---|
| SleepFM From-Scratch | 2056 | 19.682 | 5.32e-05 |
| SleepFM Next-Token | 2056 | 37.561 | 6.98e-09 |
| BIOT | 2056 | 32.152 | 1.04e-07 |


**Holm-adjusted post-hoc pairwise Wilcoxon (per model):**

*No marks: post-hoc comparisons, not a ranked comparison.*
<!-- spec: items=post-hoc pairwise comparisons; marks=none; note=post-hoc comparisons, not a ranked comparison -->
| Model | Condition A | Condition B | n subjects | Median diff macro F1 | Wilcoxon p | Rank-biserial | Holm p |
|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only | 2056 | -0.0051 | 0.0001 | -0.099 | 0.0003 |
| SleepFM From-Scratch | EEG1 only | EEG3 only | 2056 | 0.0022 | 0.4594 | 0.019 | 0.4594 |
| SleepFM From-Scratch | EEG2 only | EEG3 only | 2056 | 0.0058 | 0.0099 | 0.066 | 0.0198 |
| SleepFM Next-Token | EEG1 only | EEG2 only | 2056 | -0.0029 | 0.0480 | -0.050 | 0.0480 |
| SleepFM Next-Token | EEG1 only | EEG3 only | 2056 | 0.0052 | 0.0037 | 0.074 | 0.0074 |
| SleepFM Next-Token | EEG2 only | EEG3 only | 2056 | 0.0076 | 1.38e-05 | 0.111 | 4.15e-05 |
| BIOT | EEG1 only | EEG2 only | 2056 | 0.0051 | 0.0010 | 0.084 | 0.0020 |
| BIOT | EEG1 only | EEG3 only | 2056 | -0.0006 | 0.1587 | -0.036 | 0.1587 |
| BIOT | EEG2 only | EEG3 only | 2056 | -0.0071 | 3.97e-07 | -0.129 | 1.19e-06 |


### D.5: per-class recall
**N1, N3, REM:**

*No marks: per-class comparisons, not a ranked comparison.*
<!-- spec: items=per-class recall comparisons; marks=none; note=per-class comparisons, not a ranked comparison -->
| Model | Stage | Condition A | Condition B | Reference | n subjects | n dropped (undefined) | Median diff recall | Wilcoxon p | Rank-biserial | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | N1 | EEG1 only | EEG2 only |  | 2035 | 21 | -0.0312 | 6.94e-43 | -0.362 | 2.08e-42 |
| SleepFM From-Scratch | N1 | EEG1 only | EEG3 only |  | 2035 | 21 | -0.0253 | 9.61e-19 | -0.231 | 1.92e-18 |
| SleepFM From-Scratch | N1 | EEG2 only | EEG3 only |  | 2035 | 21 | 0.0098 | 0.0164 | 0.063 | 0.0164 |
| SleepFM From-Scratch | N1 | EEG1 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.1111 | 3.01e-232 | -0.847 | 1.81e-231 |
| SleepFM From-Scratch | N1 | EEG2 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0714 | 2.77e-171 | -0.731 | 1.39e-170 |
| SleepFM From-Scratch | N1 | EEG3 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0781 | 9.99e-153 | -0.687 | 3.99e-152 |
| SleepFM From-Scratch | N3 | EEG1 only | EEG2 only |  | 1885 | 171 | 0.0000 | 4.10e-06 | -0.135 | 1.23e-05 |
| SleepFM From-Scratch | N3 | EEG1 only | EEG3 only |  | 1885 | 171 | 0.0000 | 0.0206 | 0.067 | 0.0206 |
| SleepFM From-Scratch | N3 | EEG2 only | EEG3 only |  | 1885 | 171 | 0.0000 | 1.31e-09 | 0.175 | 5.25e-09 |
| SleepFM From-Scratch | N3 | EEG1 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | 0.0000 | 1.21e-13 | -0.219 | 6.06e-13 |
| SleepFM From-Scratch | N3 | EEG2 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | 0.0000 | 0.0024 | -0.090 | 0.0048 |
| SleepFM From-Scratch | N3 | EEG3 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | -0.0085 | 5.52e-20 | -0.268 | 3.31e-19 |
| SleepFM From-Scratch | REM | EEG1 only | EEG2 only |  | 2012 | 44 | 0.0000 | 0.8624 | 0.005 | 1.0000 |
| SleepFM From-Scratch | REM | EEG1 only | EEG3 only |  | 2012 | 44 | 0.0000 | 0.2816 | 0.031 | 0.8448 |
| SleepFM From-Scratch | REM | EEG2 only | EEG3 only |  | 2012 | 44 | 0.0000 | 0.6125 | 0.014 | 1.0000 |
| SleepFM From-Scratch | REM | EEG1 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | 0.0000 | 2.72e-16 | -0.239 | 1.09e-15 |
| SleepFM From-Scratch | REM | EEG2 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | 0.0000 | 7.00e-22 | -0.280 | 3.50e-21 |
| SleepFM From-Scratch | REM | EEG3 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | 0.0000 | 2.91e-24 | -0.295 | 1.75e-23 |
| SleepFM Next-Token | N1 | EEG1 only | EEG2 only |  | 2035 | 21 | -0.0122 | 2.89e-05 | -0.111 | 2.89e-05 |
| SleepFM Next-Token | N1 | EEG1 only | EEG3 only |  | 2035 | 21 | 0.0364 | 2.12e-35 | 0.326 | 4.25e-35 |
| SleepFM Next-Token | N1 | EEG2 only | EEG3 only |  | 2035 | 21 | 0.0476 | 2.82e-57 | 0.417 | 8.47e-57 |
| SleepFM Next-Token | N1 | EEG1 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0441 | 2.64e-77 | -0.490 | 1.32e-76 |
| SleepFM Next-Token | N1 | EEG2 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0303 | 1.34e-58 | -0.427 | 5.37e-58 |
| SleepFM Next-Token | N1 | EEG3 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0741 | 8.83e-171 | -0.729 | 5.30e-170 |
| SleepFM Next-Token | N3 | EEG1 only | EEG2 only |  | 1885 | 171 | 0.0000 | 2.57e-07 | -0.153 | 1.03e-06 |
| SleepFM Next-Token | N3 | EEG1 only | EEG3 only |  | 1885 | 171 | 0.0000 | 8.10e-05 | -0.117 | 0.0002 |
| SleepFM Next-Token | N3 | EEG2 only | EEG3 only |  | 1885 | 171 | 0.0000 | 0.7503 | -0.009 | 1.0000 |
| SleepFM Next-Token | N3 | EEG1 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | 0.0000 | 0.8223 | 0.007 | 1.0000 |
| SleepFM Next-Token | N3 | EEG2 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | 0.0000 | 4.70e-10 | 0.188 | 2.35e-09 |
| SleepFM Next-Token | N3 | EEG3 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | 0.0000 | 3.22e-11 | 0.199 | 1.93e-10 |
| SleepFM Next-Token | REM | EEG1 only | EEG2 only |  | 2012 | 44 | 0.0000 | 0.8006 | -0.007 | 1.0000 |
| SleepFM Next-Token | REM | EEG1 only | EEG3 only |  | 2012 | 44 | 0.0000 | 0.0005 | 0.100 | 0.0022 |
| SleepFM Next-Token | REM | EEG2 only | EEG3 only |  | 2012 | 44 | 0.0000 | 1.64e-05 | 0.125 | 8.20e-05 |
| SleepFM Next-Token | REM | EEG1 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | 0.0000 | 0.3476 | -0.028 | 1.0000 |
| SleepFM Next-Token | REM | EEG2 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | 0.0000 | 0.3690 | -0.027 | 1.0000 |
| SleepFM Next-Token | REM | EEG3 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | 0.0000 | 1.01e-07 | -0.159 | 6.08e-07 |
| BIOT | N1 | EEG1 only | EEG2 only |  | 2035 | 21 | 0.0471 | 1.11e-52 | 0.399 | 3.34e-52 |
| BIOT | N1 | EEG1 only | EEG3 only |  | 2035 | 21 | 0.0000 | 0.6598 | 0.011 | 0.6598 |
| BIOT | N1 | EEG2 only | EEG3 only |  | 2035 | 21 | -0.0470 | 6.84e-33 | -0.312 | 1.37e-32 |
| BIOT | N1 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2035 | 21 | -0.0753 | 1.63e-136 | -0.651 | 6.50e-136 |
| BIOT | N1 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2035 | 21 | -0.1250 | 2.49e-228 | -0.835 | 1.50e-227 |
| BIOT | N1 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2035 | 21 | -0.0698 | 2.11e-143 | -0.670 | 1.05e-142 |
| BIOT | N3 | EEG1 only | EEG2 only |  | 1885 | 171 | 0.0000 | 5.08e-13 | -0.208 | 2.03e-12 |
| BIOT | N3 | EEG1 only | EEG3 only |  | 1885 | 171 | -0.0256 | 9.77e-27 | -0.307 | 5.86e-26 |
| BIOT | N3 | EEG2 only | EEG3 only |  | 1885 | 171 | 0.0000 | 5.19e-10 | -0.181 | 1.56e-09 |
| BIOT | N3 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 1885 | 171 | 0.0000 | 1.87e-07 | -0.152 | 3.74e-07 |
| BIOT | N3 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 1885 | 171 | 0.0000 | 0.1650 | 0.041 | 0.1650 |
| BIOT | N3 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 1885 | 171 | 0.0088 | 1.56e-22 | 0.288 | 7.81e-22 |
| BIOT | REM | EEG1 only | EEG2 only |  | 2012 | 44 | 0.0062 | 7.49e-05 | 0.104 | 0.0002 |
| BIOT | REM | EEG1 only | EEG3 only |  | 2012 | 44 | 0.0000 | 0.0145 | 0.064 | 0.0289 |
| BIOT | REM | EEG2 only | EEG3 only |  | 2012 | 44 | 0.0000 | 0.1532 | -0.037 | 0.1532 |
| BIOT | REM | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2012 | 44 | -0.0398 | 5.17e-65 | -0.448 | 2.07e-64 |
| BIOT | REM | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2012 | 44 | -0.0444 | 4.27e-82 | -0.505 | 2.56e-81 |
| BIOT | REM | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2012 | 44 | -0.0392 | 2.45e-75 | -0.484 | 1.22e-74 |

Holm adjustment is applied within each (model, stage) pair across its 6 comparisons. A subject is dropped from a pair when that stage is undefined (absent from that subject's night) in either condition being compared; `scipy.stats.wilcoxon(zero_method="wilcox")` additionally drops exact-zero paired differences before ranking, which is why a row can show `median_diff_recall=0.0000` together with a significant p-value when the nonzero subset drives the result.

**Wake, N2:**

*No marks: per-class comparisons, not a ranked comparison.*
<!-- spec: items=per-class recall comparisons; marks=none; note=per-class comparisons, not a ranked comparison -->
| Model | Stage | Condition A | Condition B | Reference | n subjects | n dropped (undefined) | Median diff recall | Wilcoxon p | Rank-biserial | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | Wake | EEG1 only | EEG2 only |  | 2056 | 0 | -0.0048 | 1.82e-33 | -0.316 | 5.47e-33 |
| SleepFM From-Scratch | Wake | EEG1 only | EEG3 only |  | 2056 | 0 | -0.0042 | 7.31e-24 | -0.262 | 1.46e-23 |
| SleepFM From-Scratch | Wake | EEG2 only | EEG3 only |  | 2056 | 0 | 0.0000 | 0.5159 | 0.017 | 0.5159 |
| SleepFM From-Scratch | Wake | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0107 | 6.74e-145 | -0.672 | 4.05e-144 |
| SleepFM From-Scratch | Wake | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0045 | 1.11e-58 | -0.428 | 4.45e-58 |
| SleepFM From-Scratch | Wake | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0051 | 1.93e-60 | -0.431 | 9.67e-60 |
| SleepFM From-Scratch | N2 | EEG1 only | EEG2 only |  | 2056 | 0 | 0.0284 | 1.43e-40 | 0.343 | 8.58e-40 |
| SleepFM From-Scratch | N2 | EEG1 only | EEG3 only |  | 2056 | 0 | 0.0299 | 6.98e-23 | 0.252 | 2.79e-22 |
| SleepFM From-Scratch | N2 | EEG2 only | EEG3 only |  | 2056 | 0 | -0.0040 | 0.1016 | -0.042 | 0.1016 |
| SleepFM From-Scratch | N2 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | 0.0093 | 0.0065 | 0.070 | 0.0129 |
| SleepFM From-Scratch | N2 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0173 | 3.60e-28 | -0.282 | 1.80e-27 |
| SleepFM From-Scratch | N2 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0178 | 7.55e-22 | -0.246 | 2.27e-21 |
| SleepFM Next-Token | Wake | EEG1 only | EEG2 only |  | 2056 | 0 | -0.0031 | 1.53e-11 | -0.177 | 3.06e-11 |
| SleepFM Next-Token | Wake | EEG1 only | EEG3 only |  | 2056 | 0 | -0.0064 | 8.69e-31 | -0.301 | 3.47e-30 |
| SleepFM Next-Token | Wake | EEG2 only | EEG3 only |  | 2056 | 0 | -0.0031 | 8.90e-12 | -0.179 | 2.67e-11 |
| SleepFM Next-Token | Wake | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0072 | 2.49e-73 | -0.475 | 1.49e-72 |
| SleepFM Next-Token | Wake | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0031 | 1.51e-34 | -0.325 | 7.56e-34 |
| SleepFM Next-Token | Wake | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | 0.0000 | 0.0047 | -0.075 | 0.0047 |
| SleepFM Next-Token | N2 | EEG1 only | EEG2 only |  | 2056 | 0 | 0.0142 | 1.12e-12 | 0.182 | 3.35e-12 |
| SleepFM Next-Token | N2 | EEG1 only | EEG3 only |  | 2056 | 0 | 0.0035 | 0.4005 | 0.022 | 0.4005 |
| SleepFM Next-Token | N2 | EEG2 only | EEG3 only |  | 2056 | 0 | -0.0107 | 9.67e-08 | -0.136 | 1.93e-07 |
| SleepFM Next-Token | N2 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0191 | 7.11e-40 | -0.339 | 2.84e-39 |
| SleepFM Next-Token | N2 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0349 | 8.13e-100 | -0.544 | 4.88e-99 |
| SleepFM Next-Token | N2 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0218 | 6.84e-46 | -0.365 | 3.42e-45 |
| BIOT | Wake | EEG1 only | EEG2 only |  | 2056 | 0 | -0.0082 | 2.48e-19 | -0.232 | 7.44e-19 |
| BIOT | Wake | EEG1 only | EEG3 only |  | 2056 | 0 | -0.0062 | 1.49e-12 | -0.182 | 2.99e-12 |
| BIOT | Wake | EEG2 only | EEG3 only |  | 2056 | 0 | 0.0015 | 0.0625 | 0.048 | 0.0625 |
| BIOT | Wake | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0140 | 9.55e-99 | -0.544 | 5.73e-98 |
| BIOT | Wake | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0052 | 4.52e-33 | -0.311 | 1.81e-32 |
| BIOT | Wake | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0078 | 4.32e-52 | -0.392 | 2.16e-51 |
| BIOT | N2 | EEG1 only | EEG2 only |  | 2056 | 0 | 0.0036 | 0.0157 | 0.062 | 0.0157 |
| BIOT | N2 | EEG1 only | EEG3 only |  | 2056 | 0 | 0.0135 | 5.80e-07 | 0.128 | 1.74e-06 |
| BIOT | N2 | EEG2 only | EEG3 only |  | 2056 | 0 | 0.0105 | 0.0068 | 0.069 | 0.0135 |
| BIOT | N2 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0439 | 2.10e-128 | -0.617 | 1.05e-127 |
| BIOT | N2 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0468 | 7.18e-128 | -0.616 | 2.87e-127 |
| BIOT | N2 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0591 | 1.43e-142 | -0.651 | 8.59e-142 |


### D.6: models against each other, same subjects, per condition

*No marks: head-to-head comparisons, not a ranked comparison.*
<!-- spec: items=model-vs-model comparisons; marks=none; note=head-to-head comparisons, not a ranked comparison -->
| Model A | Model B | Condition | n subjects | Median diff macro F1 (A - B) | Wilcoxon p | Rank-biserial | Patience / pretraining note |
|---|---|---|---|---|---|---|---|
| BIOT | SleepFM Next-Token | EEG1 only | 2056 | -0.0072 | 1.41e-13 | -0.188 | BIOT patience 20 (EEG, 3-channel benchmark) vs. patience 10 (single-channel); Next-Token is pretrained, BIOT is a published pretrained foundation model |
| BIOT | SleepFM Next-Token | EEG2 only | 2056 | -0.0127 | 1.34e-28 | -0.282 | BIOT patience 20 (EEG, 3-channel benchmark) vs. patience 10 (single-channel); Next-Token is pretrained, BIOT is a published pretrained foundation model |
| BIOT | SleepFM Next-Token | EEG3 only | 2056 | -0.0012 | 0.0836 | -0.044 | BIOT patience 20 (EEG, 3-channel benchmark) vs. patience 10 (single-channel); Next-Token is pretrained, BIOT is a published pretrained foundation model |
| BIOT | SleepFM Next-Token | 3-channel benchmark | 2056 | 0.0135 | 9.63e-23 | 0.250 | BIOT patience 20 (EEG, 3-channel benchmark) vs. patience 10 (single-channel); Next-Token is pretrained, BIOT is a published pretrained foundation model |
| BIOT | SleepFM From-Scratch | EEG1 only | 2056 | 0.0140 | 1.01e-12 | 0.182 | BIOT patience 20 (EEG, 3-channel benchmark) vs. patience 10 (single-channel); From-Scratch has no pretraining, BIOT is a published pretrained foundation model |
| BIOT | SleepFM From-Scratch | EEG2 only | 2056 | 0.0032 | 0.3907 | 0.022 | BIOT patience 20 (EEG, 3-channel benchmark) vs. patience 10 (single-channel); From-Scratch has no pretraining, BIOT is a published pretrained foundation model |
| BIOT | SleepFM From-Scratch | EEG3 only | 2056 | 0.0126 | 9.94e-18 | 0.218 | BIOT patience 20 (EEG, 3-channel benchmark) vs. patience 10 (single-channel); From-Scratch has no pretraining, BIOT is a published pretrained foundation model |
| BIOT | SleepFM From-Scratch | 3-channel benchmark | 2056 | 0.0105 | 4.36e-18 | 0.221 | BIOT patience 20 (EEG, 3-channel benchmark) vs. patience 10 (single-channel); From-Scratch has no pretraining, BIOT is a published pretrained foundation model |
| SleepFM From-Scratch | SleepFM Next-Token | EEG1 only | 2056 | -0.0184 | 3.23e-64 | -0.431 | both SleepFM, same patience per condition; differ only in pretraining objective |
| SleepFM From-Scratch | SleepFM Next-Token | EEG2 only | 2056 | -0.0174 | 6.61e-53 | -0.390 | both SleepFM, same patience per condition; differ only in pretraining objective |
| SleepFM From-Scratch | SleepFM Next-Token | EEG3 only | 2056 | -0.0140 | 4.46e-36 | -0.319 | both SleepFM, same patience per condition; differ only in pretraining objective |
| SleepFM From-Scratch | SleepFM Next-Token | 3-channel, patience 10 | 2056 | -0.0066 | 7.49e-10 | -0.157 | both SleepFM, same patience per condition; differ only in pretraining objective |
| SleepFM From-Scratch | SleepFM Next-Token | 3-channel benchmark | 2056 | 0.0023 | 0.1026 | 0.042 | both SleepFM, same patience per condition; differ only in pretraining objective |


### D.7: fold-level sign counts

*No marks: fold-level differences, not a ranked comparison.*
<!-- spec: items=fold-level sign counts; marks=none; note=fold-level differences, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | n folds | Per-fold diff (macro F1, A - B) | Sign counts |
|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | 5 | 0.0099, -0.0106, -0.0176, 0.0058, 0.0064 | +3/-2/0:0 |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 5 | -0.0018, -0.0021, -0.0264, 0.0058, 0.0183 | +2/-3/0:0 |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | 5 | -0.0117, 0.0085, -0.0088, -0.0000, 0.0119 | +2/-3/0:0 |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | 5 | -0.0232, -0.0520, -0.0649, -0.0398, -0.0435 | +0/-5/0:0 |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | 5 | -0.0331, -0.0414, -0.0472, -0.0457, -0.0500 | +0/-5/0:0 |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | 5 | -0.0214, -0.0499, -0.0384, -0.0456, -0.0619 | +0/-5/0:0 |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 5 | -0.0002, 0.0044, 0.0044, -0.0070, 0.0130 | +3/-2/0:0 |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 5 | -0.0020, 0.0042, 0.0085, -0.0048, 0.0144 | +3/-2/0:0 |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 5 | -0.0017, -0.0002, 0.0041, 0.0022, 0.0014 | +3/-2/0:0 |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | 5 | -0.0260, -0.0253, -0.0341, -0.0363, -0.0302 | +0/-5/0:0 |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | 5 | -0.0257, -0.0297, -0.0386, -0.0293, -0.0432 | +0/-5/0:0 |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | 5 | -0.0240, -0.0296, -0.0426, -0.0315, -0.0446 | +0/-5/0:0 |
| BIOT | EEG1 only | EEG2 only |  | 5 | 0.0110, 0.0074, 0.0139, 0.0031, 0.0059 | +5/-0/0:0 |
| BIOT | EEG1 only | EEG3 only |  | 5 | -0.0022, -0.0044, -0.0065, -0.0074, 0.0001 | +1/-4/0:0 |
| BIOT | EEG2 only | EEG3 only |  | 5 | -0.0131, -0.0117, -0.0204, -0.0104, -0.0058 | +0/-5/0:0 |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0617, -0.0550, -0.0597, -0.0555, -0.0611 | +0/-5/0:0 |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0727, -0.0624, -0.0736, -0.0586, -0.0671 | +0/-5/0:0 |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0596, -0.0507, -0.0532, -0.0481, -0.0612 | +0/-5/0:0 |

A two-sided Wilcoxon signed-rank test on 5 paired fold values cannot reach below p = 0.0625 (2/2^5), even when all 5 folds agree in sign -- no p-value is reported at this level for that reason; only the fold-level differences and sign counts above.

## Part E: compute profile

### E1: parameters and checkpoint size

*Compared: models, down each column within each block, higher is better.*
<!-- spec: block=Condition; items=models; value_cols=Total params,Checkpoint size (MB); bold=down; green=none -->
| Model | Condition | Total params | Trainable params | Encoder params | Head params | Checkpoint size (MB) | fp16 estimate (MB) | Note |
|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | 1 channel | **7,019,179** | 2,170,629 | 4,848,550 | 2,170,629 | 45.301 | 13.388 | encoder is frozen during fine-tuning; only the head is trainable |
| SleepFM From-Scratch | 3 channels | **7,019,179** | 2,170,629 | 4,848,550 | 2,170,629 | 45.301 | 13.388 | encoder is frozen during fine-tuning; only the head is trainable |
| SleepFM Next-Token | 1 channel | **7,019,179** | 2,170,629 | 4,848,550 | 2,170,629 | **64.551** | 13.388 | encoder is frozen during fine-tuning; only the head is trainable |
| SleepFM Next-Token | 3 channels | **7,019,179** | 2,170,629 | 4,848,550 | 2,170,629 | **64.551** | 13.388 | encoder is frozen during fine-tuning; only the head is trainable |
| BIOT | 1 channel | 3,443,991 | 3,443,991 | n/a | n/a | 13.159 | 6.569 | single end-to-end checkpoint, fully fine-tuned |
| BIOT | 3 channels | 3,443,991 | 3,443,991 | n/a | n/a | 13.159 | 6.569 | single end-to-end checkpoint, fully fine-tuned |

SleepFM's SetTransformer encoder and BIOT's encoder both have channel-count-invariant parameter counts (shared per-channel weights for SleepFM; BIOT's published architecture always uses a fixed 18-channel positional embedding regardless of how many channels are actually fed in) -- this is why `total_params` does not change between 1 and 3 channels for any model here; it is a real architectural property, not an error.

### E2: FLOPs per epoch and per night

*No marks: FLOPs counts, not a ranked comparison.*
<!-- spec: items=FLOPs counts; marks=none; note=FLOPs counts, not a ranked comparison -->
| Model | Component | Condition | FLOPs / 30s epoch | FLOPs / 8h night |
|---|---|---|---|---|
| BIOT | end-to-end | 1 channel | 381,982,720 | 366,703,411,200 |
| BIOT | end-to-end | 3 channels | 1,145,943,040 | 1,100,105,318,400 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | 9,796,608 | 9,404,743,680 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | 29,389,824 | 28,214,231,040 |
| SleepFM (From-Scratch/Next-Token share this architecture) | head | n/a (operates on embeddings, whole night at once) | 1,280 | 1,228,800 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder+head together | 1 channel | 9,797,888 | 9,405,972,480 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder+head together | 3 channels | 29,391,104 | 28,215,459,840 |

Counted with `torch.utils.flop_counter.FlopCounterMode`: matmuls and convolutions (including the internal matmuls of attention), both forward passes only (inference, no backward). Excludes elementwise ops, normalization, and BIOT's `torch.stft` front end (not a matmul/conv). SleepFM's head operates once per whole night on precomputed per-epoch embeddings (not per 30s epoch at inference time in the same sense as the encoder), so its "per epoch" FLOPs are reported as that night-level cost divided by the night's epoch count, for comparability; "encoder+head together" sums the two. BIOT's end-to-end FLOPs/epoch is about 39x the SleepFM encoder's at both 1 channel (381,982,720 / 9,796,608 = 39.0x) and 3 channels (1,145,943,040 / 29,389,824 = 39.0x).

### E3: latency, throughput, and peak GPU memory
Run as one short job per node type (45-minute limit each), 10 warm-up + 100 timed runs per device/batch combination, CUDA-synchronized, median and 95th percentile reported. **Deviation from the task's "repeat 3 times": `compute_profile.py`'s `run_benchmark()` ran this 10+100 pattern once per device, not three separate repeats -- noted here as a shortfall against the spec, not papered over; the single-run median/p95 below should be read with that in mind.** Excludes data loading and preprocessing (the HDF5/npz read and any normalization) in both cases -- only the model forward pass is timed. **Also note:** `run_benchmark()` always feeds a 3-channel input (BIOT's `(b, 3, 6000)`, SleepFM encoder's `(b, 3, 3840)`) and never a separate 1-channel shape, so the GPU batch=128 latency figure quoted again in E6 is the same number for both the "1 channel" and "3 channels" rows there -- a real gap against a literal per-channel-count reading of the task, though consistent with both models' channel-count-invariant parameter counts (E1).

**Shared-node:**

*No marks: latency benchmark, not a ranked comparison.*
<!-- spec: items=latency benchmark (shared-node); marks=none; note=latency benchmark, not a ranked comparison -->
| Model | Component | Device | Median latency (ms) | p95 latency (ms) | Throughput (epochs/s) | Peak GPU memory at batch 128 (MB) |
|---|---|---|---|---|---|---|
| BIOT | end-to-end | GPU batch=1 | 3.501 | 5.524 | 285.6 | n/a |
| BIOT | end-to-end | GPU batch=16 | 3.995 | 6.120 | 4004.5 | n/a |
| BIOT | end-to-end | GPU batch=128 | 6.980 | 7.678 | 18339.4 | 335.178 |
| BIOT | end-to-end | CPU threads=1 | 248.927 | 249.320 | 64.3 | n/a |
| BIOT | end-to-end | CPU threads=8 | 207.867 | 208.258 | 77.0 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | GPU batch=1 | 2.818 | 2.930 | 354.9 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | GPU batch=16 | 3.181 | 3.242 | 5029.9 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | GPU batch=128 | 3.138 | 3.205 | 40793.3 | 104.180 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | CPU threads=1 | 42.582 | 42.782 | 375.7 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | CPU threads=8 | 29.253 | 29.510 | 546.9 | n/a |

Node: `rg2127`. `squeue -w <node>`: (unavailable on this node: [Errno 2] No such file or directory: 'squeue') GPU before: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 1 MiB, 97871 MiB`. GPU after: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 1257 MiB, 97871 MiB`.

**Exclusive-node:** an exclusive-node probe job returned in about 3 seconds (well under the 12-hour fallback threshold checked beforehand), so the real exclusive-node benchmark below was run opportunistically in parallel with Part C, in addition to -- not instead of -- the shared-node run above, without holding up Parts A-D/F.

*No marks: latency benchmark, not a ranked comparison.*
<!-- spec: items=latency benchmark (exclusive-node); marks=none; note=latency benchmark, not a ranked comparison -->
| Model | Component | Device | Median latency (ms) | p95 latency (ms) | Throughput (epochs/s) | Peak GPU memory at batch 128 (MB) |
|---|---|---|---|---|---|---|
| BIOT | end-to-end | GPU batch=1 | 4.120 | 4.254 | 242.7 | n/a |
| BIOT | end-to-end | GPU batch=16 | 3.719 | 7.760 | 4302.7 | n/a |
| BIOT | end-to-end | GPU batch=128 | 6.983 | 7.026 | 18331.3 | 335.178 |
| BIOT | end-to-end | CPU threads=1 | 251.904 | 252.441 | 63.5 | n/a |
| BIOT | end-to-end | CPU threads=8 | 207.331 | 207.740 | 77.2 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | GPU batch=1 | 3.265 | 3.414 | 306.3 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | GPU batch=16 | 3.702 | 3.799 | 4322.3 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | GPU batch=128 | 3.633 | 3.703 | 35230.0 | 104.180 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | CPU threads=1 | 43.106 | 43.225 | 371.2 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | CPU threads=8 | 30.022 | 30.257 | 532.9 | n/a |

Node: `rg1113`. `squeue -w <node>`: (unavailable on this node: [Errno 2] No such file or directory: 'squeue') GPU before: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 4 MiB, 97871 MiB
1, 0 %, 0 %, 3 MiB, 97871 MiB
2, 0 %, 0 %, 4 MiB, 97871 MiB
3, 0 %, 0 %, 3 MiB, 97871 MiB`. GPU after: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 1259 MiB, 97871 MiB
1, 0 %, 0 %, 3 MiB, 97871 MiB
2, 0 %, 0 %, 4 MiB, 97871 MiB
3, 0 %, 0 %, 3 MiB, 97871 MiB`.

`squeue`/co-tenancy could not be queried directly from inside either job (`squeue` is not on the compute node's PATH); GPU memory/utilization before and after, and the node's hostname, were captured via `nvidia-smi` instead and are the only direct signal available for shared-vs-exclusive co-tenancy here. The shared-node and exclusive-node numbers above are close throughout (within benchmarking noise), consistent with this specific shared node having had little contending load at measurement time.

### E4: input data size per channel per night

*No marks: input size, not a ranked comparison.*
<!-- spec: items=input data size; marks=none; note=input size, not a ranked comparison -->
| Model | Sampling rate (Hz) | Samples / channel / night | Bytes / channel / night | MB / channel / night |
|---|---|---|---|---|
| SleepFM From-Scratch | 128 | 3,686,400 | 14,745,600 | 14.062 |
| SleepFM Next-Token | 128 | 3,686,400 | 14,745,600 | 14.062 |
| BIOT | 200 | 5,760,000 | 23,040,000 | 21.973 |


### E5: training cost from sacct
**Not recoverable.** See the E5 note under "Summary of tests" above: `sacct` retains no records before approximately 2026-11-11 on this cluster, and no surviving log or config field records the actual stop epoch or best epoch for the original (July-October 2026) pretraining and fine-tuning jobs in this task's scope. No numbers are substituted in their place.

### E6: accuracy against compute

*Compared: models, down each column within each block, higher is better.*
<!-- spec: block=Condition; items=models; value_cols=Pooled macro F1 (95% CI); bold=down; green=none -->
| Model | Condition | Pooled macro F1 (95% CI) | Total params | FLOPs / epoch (end-to-end) | GPU batch=128 median latency (ms, shared-node) |
|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG3 only | 0.6427 [0.6373, 0.6481] | 7,019,179 | 9,797,888 | 3.138 |
| SleepFM From-Scratch | 3-channel benchmark | 0.6984 [0.6938, 0.7031] | 7,019,179 | 29,391,104 | 3.138 |
| SleepFM Next-Token | EEG3 only | **0.6584 [0.6528, 0.6642]** | 7,019,179 | 9,797,888 | 3.138 |
| SleepFM Next-Token | 3-channel benchmark | 0.6960 [0.6914, 0.7009] | 7,019,179 | 29,391,104 | 3.138 |
| BIOT | EEG3 only | 0.6538 [0.6479, 0.6596] | 3,443,991 | 381,982,720 | 6.980 |
| BIOT | 3-channel benchmark | **0.7084 [0.7036, 0.7130]** | 3,443,991 | 1,145,943,040 | 6.980 |

"1 channel" uses EEG3 (the C4-M1 electrode match to MESA's own lead) as the representative single channel; "3 channels" uses each model's 3-channel benchmark run. SleepFM's `flops_per_epoch` here is "encoder+head together" (see E2's note on how the head's per-night cost is divided into a per-epoch figure).

## GPU-hours used
All 8 GPU jobs run for this task (`sacct -u hamehd --starttime 2026-11-10`, excluding `.batch`/`.extern` sub-steps). Part A, C, E1, and E4 needed no GPU and ran on a login-node CPU venv, so they do not appear here.

*No marks: job list, GPU-hours summed below.*
<!-- spec: items=job list; marks=none; note=job list, GPU-hours summed below -->
| Job ID | Purpose | Elapsed | GPUs |
|---|---|---|---|
| 3084148 | BIOT eval rerun (first attempt, pre-incident) | 01:15:05 | 1 |
| 3092867 | BIOT eval rerun (clean, verified, --no_checkpoint_link) | 01:10:45 | 1 |
| 3093075 | Part E2 FLOPs (first attempt) | 00:00:10 | 1 |
| 3093089 | Part E3 exclusive-node probe (returned in ~3s) | 00:00:03 | 4 |
| 3093108 | Part E3 shared-node (first attempt) | 00:00:07 | 1 |
| 3093150 | Part E3 shared-node (real benchmark, used above) | 00:01:15 | 1 |
| 3093319 | Part E2 FLOPs (canonical run, used above) | 00:00:10 | 1 |
| 3094289 | Part E3 exclusive-node (real benchmark, used above) | 00:01:17 | 4 |

Total: (4505 + 4245 + 10 + 3x4 + 7 + 75 + 10 + 77x4) GPU-seconds = 9172 GPU-seconds = **2.55 GPU-hours**, almost entirely the two full BIOT evaluation reruns (01:15:05 and 01:10:45 on 1 GPU each); every Part E job ran in under 2 minutes.

