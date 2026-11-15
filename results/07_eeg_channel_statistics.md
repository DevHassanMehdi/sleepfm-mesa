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
| Part E5: training cost from sacct | recovered, 275.6 GPU-hours across 82 rows -- see note below |
| `scripts/results_tables.py --check` | 0 issues (run after --fix, see Part F commit) |
| `scripts/check_results_numbers.py` | run after --fix; see commit |
| Unit tests (test_channel_stats.py, test_compute_profile.py, test_experiment_paths.py, full suite) | 62 passed |
| Link check | 0 broken links |

**E5 note:** the Phase 1 report's original conclusion -- that training cost was unrecoverable because `sacct` has no records before approximately 2026-11-11 -- was querying the wrong account. `sacct -S 2026-07-01 -u hamehdi` (the pre-migration account these jobs actually ran under) returns full historical accounting data back to July 2026. Training cost for the pretraining, CPU-cache, embedding, and fine-tuning jobs in this task's scope is now in Part E5, with GPU-hours confirmed via sacct for every row. Epoch counts (stop/best) are also recovered for most rows via per-job `.log` files or (for the two pretraining runs) the checkpoint's own `epoch` field -- except for the SleepFM and BIOT 3-channel-benchmark fine-tuning runs, whose logs sit under the old account's unreadable home directory; Part E5 gives the exact recovery command for those.

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
Redone from the first version of this report, which (a) fed a fixed 3-channel input to both models regardless of the labeled condition, (b) never benchmarked the SleepFM staging head, and (c) ran the 10-warmup/100-timed pattern once instead of three repeats. Now: BIOT is fed its own `(B, 1, 6000)` / `(B, 3, 6000)` per condition; the SleepFM encoder is fed `(B, 1, 3840)` / `(B, 3, 3840)` per condition; the staging head is benchmarked separately on a whole night of embeddings (`n_ch=3`, the shape it is actually trained/used with -- see E2's note). Every (model, condition, device) cell is measured 3 independent times, each with its own 10 warm-up and 100 timed calls, CUDA-synchronized around every call; the 3 repeats' medians/p95s are listed separately below, not averaged away. Excludes data loading and preprocessing -- only the model forward pass is timed. GPU: exclusive GH200 node (`gpumedium`, `--exclusive`, 45-minute limit). CPU: exclusive node on the `medium` CPU partition (`--exclusive`, 45-minute limit) -- the `small` partition rejects `--exclusive` ("administratively prohibited on shared partitions"); `medium` accepts it. The shared-node run (no `--exclusive`, normal `gpumedium`) is kept alongside, labeled.

**Shared-node (GPU + CPU together, `gpumedium`, no `--exclusive`):**

*No marks: latency benchmark, not a ranked comparison.*
<!-- spec: items=latency benchmark (shared-node); marks=none; note=latency benchmark, not a ranked comparison -->
| Model | Component | Condition | Device | Median latency (ms, 3 repeats) | p95 latency (ms, 3 repeats) | Throughput (epochs/s, 3 repeats) | Peak GPU memory at batch 128 (MB) |
|---|---|---|---|---|---|---|---|
| BIOT | end-to-end | 1 channel | GPU batch=1 | 2.878 / 2.860 / 2.878 | 2.978 / 2.919 / 2.941 | 347.4 / 349.6 / 347.5 | n/a |
| BIOT | end-to-end | 1 channel | GPU batch=16 | 2.905 / 2.932 / 2.924 | 2.983 / 4.807 / 4.791 | 5507.9 / 5456.6 / 5472.0 | n/a |
| BIOT | end-to-end | 1 channel | GPU batch=128 | 3.037 / 3.052 / 3.076 | 4.793 / 4.847 / 4.812 | 42144.7 / 41940.8 / 41606.2 | 154.285 |
| BIOT | end-to-end | 1 channel | CPU threads=1 | 85.599 / 85.568 / 85.545 | 85.788 / 85.745 / 85.755 | 186.9 / 187.0 / 187.0 | n/a |
| BIOT | end-to-end | 1 channel | CPU threads=8 | 72.155 / 72.047 / 71.689 | 76.105 / 75.756 / 74.889 | 221.7 / 222.1 / 223.2 | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=1 | 3.302 / 3.295 / 3.288 | 3.393 / 3.366 / 3.353 | 302.8 / 303.5 / 304.2 | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=16 | 3.407 / 3.435 / 3.396 | 5.595 / 5.638 / 5.554 | 4696.8 / 4658.5 / 4711.1 | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=128 | 7.001 / 6.952 / 6.970 | 7.565 / 7.568 / 7.567 | 18283.5 / 18412.4 / 18365.7 | 335.178 |
| BIOT | end-to-end | 3 channels | CPU threads=1 | 250.121 / 249.862 / 249.950 | 251.019 / 250.099 / 250.347 | 64.0 / 64.0 / 64.0 | n/a |
| BIOT | end-to-end | 3 channels | CPU threads=8 | 205.466 / 207.916 / 207.829 | 215.836 / 217.157 / 217.772 | 77.9 / 77.0 / 77.0 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | GPU batch=1 | 2.360 / 2.313 / 2.313 | 2.500 / 2.360 / 2.356 | 423.7 / 432.4 / 432.4 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | GPU batch=16 | 2.717 / 2.714 / 2.744 | 2.771 / 2.758 / 4.517 | 5889.2 / 5895.7 / 5831.0 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | GPU batch=128 | 2.735 / 2.720 / 2.735 | 4.511 / 4.498 / 4.485 | 46795.8 / 47062.6 / 46801.3 | 81.680 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | CPU threads=1 | 23.344 / 23.279 / 23.287 | 23.606 / 23.362 / 23.467 | 685.4 / 687.3 / 687.1 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | CPU threads=8 | 19.379 / 19.353 / 19.309 | 22.890 / 22.790 / 22.724 | 825.6 / 826.7 / 828.6 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | GPU batch=1 | 2.806 / 2.782 / 2.788 | 2.898 / 2.835 / 2.854 | 356.4 / 359.4 / 358.7 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | GPU batch=16 | 3.240 / 3.227 / 3.215 | 5.310 / 5.263 / 5.292 | 4939.0 / 4957.5 / 4976.7 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | GPU batch=128 | 3.233 / 3.214 / 3.201 | 5.260 / 5.260 / 5.228 | 39587.9 / 39827.8 / 39984.3 | 104.180 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | CPU threads=1 | 42.592 / 42.524 / 42.454 | 42.863 / 42.603 / 42.546 | 375.7 / 376.3 / 376.9 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | CPU threads=8 | 30.216 / 30.404 / 30.328 | 33.016 / 32.903 / 33.025 | 529.5 / 526.3 / 527.6 | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=1 | 2.928 / 2.910 / 2.907 | 2.988 / 2.937 / 2.938 | 341.6 / 343.7 / 344.0 | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=16 | 6.418 / 6.441 / 6.430 | 6.445 / 6.537 / 6.531 | 2493.1 / 2483.9 / 2488.3 | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=128 | 35.237 / 35.253 / 35.403 | 35.938 / 35.909 / 36.029 | 3632.5 / 3630.9 / 3615.5 | 4213.954 |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | CPU threads=1 | 2092.230 / 2091.918 / 2093.617 | 2100.658 / 2102.365 / 2101.814 | 7.6 / 7.6 / 7.6 | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | CPU threads=8 | 1366.866 / 1365.596 / 1366.684 | 1389.813 / 1387.770 / 1395.329 | 11.7 / 11.7 / 11.7 | n/a |

Node: `rg1104`. `squeue -w <node>`: (unavailable on this node: [Errno 2] No such file or directory: 'squeue') GPU before: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 9 MiB, 97871 MiB`. GPU after: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 7686 MiB, 97871 MiB`.

**Exclusive-node, GPU (`gpumedium --exclusive`, 4x GH200):**

*No marks: latency benchmark, not a ranked comparison.*
<!-- spec: items=latency benchmark (exclusive-node, GPU); marks=none; note=latency benchmark, not a ranked comparison -->
| Model | Component | Condition | Device | Median latency (ms, 3 repeats) | p95 latency (ms, 3 repeats) | Throughput (epochs/s, 3 repeats) | Peak GPU memory at batch 128 (MB) |
|---|---|---|---|---|---|---|---|
| BIOT | end-to-end | 1 channel | GPU batch=1 | 3.957 / 3.908 / 3.905 | 4.068 / 3.975 / 3.969 | 252.7 / 255.9 / 256.1 | n/a |
| BIOT | end-to-end | 1 channel | GPU batch=16 | 3.994 / 3.986 / 4.005 | 4.121 / 4.040 / 4.077 | 4005.9 / 4013.6 / 3995.2 | n/a |
| BIOT | end-to-end | 1 channel | GPU batch=128 | 4.048 / 4.019 / 4.040 | 4.119 / 4.077 / 4.107 | 31617.4 / 31849.6 / 31684.5 | 154.285 |
| BIOT | end-to-end | 3 channels | GPU batch=1 | 4.565 / 4.538 / 4.527 | 4.673 / 4.607 / 4.603 | 219.1 / 220.3 / 220.9 | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=16 | 4.694 / 4.676 / 4.693 | 4.753 / 4.761 / 4.759 | 3408.9 / 3421.8 / 3409.4 | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=128 | 7.253 / 7.254 / 7.240 | 7.285 / 7.288 / 7.271 | 17647.0 / 17646.2 / 17680.7 | 335.178 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | GPU batch=1 | 2.923 / 2.869 / 2.858 | 3.130 / 2.914 / 2.902 | 342.2 / 348.5 / 349.8 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | GPU batch=16 | 3.343 / 3.317 / 3.300 | 3.400 / 3.367 / 5.445 | 4785.8 / 4824.0 / 4848.9 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | GPU batch=128 | 3.048 / 3.030 / 3.030 | 3.100 / 3.072 / 3.075 | 41990.3 / 42237.7 / 42241.0 | 81.680 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | GPU batch=1 | 3.129 / 3.124 / 3.113 | 3.193 / 3.164 / 3.161 | 319.6 / 320.1 / 321.2 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | GPU batch=16 | 3.563 / 3.552 / 3.544 | 3.606 / 3.583 / 3.614 | 4490.3 / 4503.9 / 4514.3 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | GPU batch=128 | 3.512 / 3.516 / 3.513 | 3.569 / 3.567 / 3.558 | 36446.3 / 36408.7 / 36436.5 | 104.180 |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=1 | 3.085 / 3.069 / 3.061 | 3.149 / 3.098 / 3.092 | 324.2 / 325.9 / 326.6 | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=16 | 6.438 / 6.444 / 6.444 | 6.453 / 6.474 / 6.469 | 2485.1 / 2482.9 / 2483.0 | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=128 | 34.579 / 34.605 / 34.607 | 34.846 / 34.909 / 34.931 | 3701.7 / 3698.9 / 3698.7 | 4219.580 |

Node: `rg2121`. `squeue -w <node>`: (unavailable on this node: [Errno 2] No such file or directory: 'squeue') GPU before: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 5 MiB, 97871 MiB
1, 0 %, 0 %, 5 MiB, 97871 MiB
2, 0 %, 0 %, 5 MiB, 97871 MiB
3, 0 %, 0 %, 27 MiB, 97871 MiB`. GPU after: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 100 %, 42 %, 7685 MiB, 97871 MiB
1, 0 %, 0 %, 4 MiB, 97871 MiB
2, 0 %, 0 %, 5 MiB, 97871 MiB
3, 0 %, 0 %, 27 MiB, 97871 MiB`.

**Exclusive-node, CPU (`medium --exclusive`, 384 CPUs):**

*No marks: latency benchmark, not a ranked comparison.*
<!-- spec: items=latency benchmark (exclusive-node, CPU); marks=none; note=latency benchmark, not a ranked comparison -->
| Model | Component | Condition | Device | Median latency (ms, 3 repeats) | p95 latency (ms, 3 repeats) | Throughput (epochs/s, 3 repeats) | Peak GPU memory at batch 128 (MB) |
|---|---|---|---|---|---|---|---|
| BIOT | end-to-end | 1 channel | CPU threads=1 | 69.104 / 70.353 / 69.644 | 72.208 / 73.901 / 72.533 | 231.5 / 227.4 / 229.7 | n/a |
| BIOT | end-to-end | 1 channel | CPU threads=8 | 17.316 / 16.976 / 16.966 | 20.602 / 20.808 / 20.716 | 924.0 / 942.5 / 943.0 | n/a |
| BIOT | end-to-end | 3 channels | CPU threads=1 | 202.521 / 202.468 / 202.449 | 202.954 / 203.045 / 202.640 | 79.0 / 79.0 / 79.0 | n/a |
| BIOT | end-to-end | 3 channels | CPU threads=8 | 39.288 / 39.269 / 38.692 | 39.506 / 40.632 / 39.016 | 407.2 / 407.4 / 413.5 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | CPU threads=1 | 12.542 / 12.527 / 12.503 | 12.711 / 12.607 / 12.546 | 1275.7 / 1277.3 / 1279.7 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | CPU threads=8 | 5.715 / 5.717 / 5.723 | 5.799 / 5.811 / 5.819 | 2799.6 / 2798.7 / 2795.8 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | CPU threads=1 | 21.346 / 21.212 / 21.125 | 21.427 / 21.269 / 21.179 | 749.6 / 754.3 / 757.4 | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | CPU threads=8 | 7.672 / 7.672 / 7.670 | 7.746 / 7.753 / 7.770 | 2085.4 / 2085.6 / 2086.0 | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | CPU threads=1 | 2269.021 / 2273.877 / 2278.488 | 2271.706 / 2279.420 / 2281.624 | 7.1 / 7.0 / 7.0 | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | CPU threads=8 | 598.459 / 594.915 / 591.921 | 604.498 / 601.992 / 599.822 | 26.7 / 26.9 / 27.0 | n/a |

Node: `rc4153`. `squeue -w <node>`:              JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
           3105306    medium         channel_e3_excl_cpu_v2   hamehd  R       0:05      1 rc4153
 GPU before: `(unavailable on this node: [Errno 2] No such file or directory: 'nvidia-smi')`. GPU after: `(unavailable on this node: [Errno 2] No such file or directory: 'nvidia-smi')`.

`squeue`/co-tenancy could not be queried directly from inside any of these jobs (`squeue` is not on the compute node's PATH); GPU memory/utilization before and after, and the node's hostname, were captured via `nvidia-smi` instead and are the only direct signal available for shared-vs-exclusive co-tenancy here. The shared-node and exclusive-node GPU numbers above are close throughout (within benchmarking noise), consistent with this specific shared node having had little contending load at measurement time.

### E4: input data size per channel per night

*No marks: input size, not a ranked comparison.*
<!-- spec: items=input data size; marks=none; note=input size, not a ranked comparison -->
| Model | Sampling rate (Hz) | Samples / channel / night | Bytes / channel / night | MB / channel / night |
|---|---|---|---|---|
| SleepFM From-Scratch | 128 | 3,686,400 | 14,745,600 | 14.062 |
| SleepFM Next-Token | 128 | 3,686,400 | 14,745,600 | 14.062 |
| BIOT | 200 | 5,760,000 | 23,040,000 | 21.973 |


### E5: training cost from sacct
Revised: `sacct -S 2026-07-01 -u hamehdi` (the pre-migration account) returns full historical accounting data -- the earlier "not recoverable" conclusion was querying the wrong account. Every job id below was matched to its (model, condition, fold) by its own `SubmitLine` (sacct), which names the exact config file and timestamp -- unambiguous. Epoch counts for most fine-tuning runs came from a `.log` file either recovered from `/projappl/project_2019517/archive/logs_2026-11-11.tar.gz` (extracted to a temp dir, deleted after use) or, for both pretraining runs, the `epoch` field inside the actual encoder checkpoint, cross-checked against a per-epoch training log whose filename timestamp matches the job's start time. Reproducible via `scripts/analysis/training_cost.py`.


*No marks: training-cost ledger, not a ranked comparison.*
<!-- spec: items=training-cost ledger; marks=none; note=training-cost ledger, not a ranked comparison -->
| Step | Model | Condition | Fold | Job ID(s) | Elapsed | GPUs | GPU-hours | Stop epoch (log) | Best epoch (log) | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| Pretrain | SleepFM From-Scratch | n/a (shared across 1ch/3ch) |  | 548808 | 6:01:30 | 1 | 6.025 | 99 | 98 | sacct (hamehdi) + checkpoint 'epoch' field |
| Pretrain | SleepFM Next-Token | n/a (shared across 1ch/3ch) |  | 539096 | 29:00:02 | 1 | 29.001 | 38 | 28 | sacct (hamehdi) + checkpoint 'epoch' field |
| CPU cache | SleepFM Next-Token | Next-Token spectral cache (CPU, prerequisite for token cache) |  | 539038 | 17:19:35 | 0 | 0.000 | n/a | n/a | sacct (hamehdi) |
| CPU cache | SleepFM Next-Token | Next-Token token cache (CPU, k-means) |  | 539095 | 0:01:49 | 0 | 0.000 | n/a | n/a | sacct (hamehdi) |
| Embed | SleepFM From-Scratch | EEG1 only |  | 2579548 | 0:10:15 | 1 | 0.171 | n/a | n/a | sacct (hamehdi); n/a, inference only |
| Embed | SleepFM From-Scratch | EEG2 only |  | 2579549 | 0:10:31 | 1 | 0.175 | n/a | n/a | sacct (hamehdi); n/a, inference only |
| Embed | SleepFM From-Scratch | EEG3 only |  | 2579550 | 0:10:04 | 1 | 0.168 | n/a | n/a | sacct (hamehdi); n/a, inference only |
| Embed | SleepFM Next-Token | EEG1 only |  | 2579551 | 0:12:04 | 1 | 0.201 | n/a | n/a | sacct (hamehdi); n/a, inference only |
| Embed | SleepFM Next-Token | EEG2 only |  | 2579552 | 0:11:02 | 1 | 0.184 | n/a | n/a | sacct (hamehdi); n/a, inference only |
| Embed | SleepFM Next-Token | EEG3 only |  | 2579553 | 0:10:43 | 1 | 0.179 | n/a | n/a | sacct (hamehdi); n/a, inference only |
| Embed | SleepFM From-Scratch | 3-channel benchmark |  | 1418746 | 0:14:44 | 1 | 0.246 | n/a | n/a | sacct (hamehdi); n/a, inference only |
| Embed | SleepFM Next-Token | 3-channel benchmark |  | 1390165 | 0:15:21 | 1 | 0.256 | n/a | n/a | sacct (hamehdi); n/a, inference only |
| Fine-tune | SleepFM From-Scratch | EEG1 only | 0 | 2580321 | 0:09:38 | 1 | 0.161 | 89 | 79 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG1 only | 1 | 2580963 | 0:05:41 | 1 | 0.095 | 60 | 50 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG1 only | 2 | 2580964 | 0:03:25 | 1 | 0.057 | 34 | 24 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG1 only | 3 | 2580965 | 0:08:34 | 1 | 0.143 | 91 | 81 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG1 only | 4 | 2580966 | 0:07:13 | 1 | 0.120 | 66 | 56 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG2 only | 0 | 2580967 | 0:06:13 | 1 | 0.104 | 52 | 42 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG2 only | 1 | 2580968 | 0:06:33 | 1 | 0.109 | 65 | 55 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG2 only | 2 | 2580969 | 0:05:09 | 1 | 0.086 | 51 | 41 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG2 only | 3 | 2580970 | 0:05:08 | 1 | 0.086 | 50 | 40 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG2 only | 4 | 2580971 | 0:05:18 | 1 | 0.088 | 55 | 45 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG3 only | 0 | 2580972 | 0:08:31 | 1 | 0.142 | 88 | 78 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG3 only | 1 | 2580973 | 0:05:54 | 1 | 0.098 | 51 | 41 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG3 only | 2 | 2580974 | 0:09:19 | 1 | 0.155 | 87 | 77 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG3 only | 3 | 2580975 | 0:05:01 | 1 | 0.084 | 50 | 40 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | EEG3 only | 4 | 2580976 | 0:04:14 | 1 | 0.071 | 34 | 24 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | 3-channel, patience 10 | 0 | 2613261 | 0:04:49 | 1 | 0.080 | 38 | 28 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | 3-channel, patience 10 | 1 | 2613262 | 0:07:20 | 1 | 0.122 | 74 | 64 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | 3-channel, patience 10 | 2 | 2613263 | 0:05:47 | 1 | 0.096 | 59 | 49 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | 3-channel, patience 10 | 3 | 2613264 | 0:05:02 | 1 | 0.084 | 48 | 38 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | 3-channel, patience 10 | 4 | 2613265 | 0:06:52 | 1 | 0.114 | 57 | 47 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM From-Scratch | 3-channel benchmark | 0 | 1418927 | 0:10:06 | 1 | 0.168 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM From-Scratch | 3-channel benchmark | 1 | 1418928 | 0:10:07 | 1 | 0.169 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM From-Scratch | 3-channel benchmark | 2 | 1418929 | 0:17:05 | 1 | 0.285 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM From-Scratch | 3-channel benchmark | 3 | 1418930 | 0:14:51 | 1 | 0.247 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM From-Scratch | 3-channel benchmark | 4 | 1418931 | 0:13:56 | 1 | 0.232 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM Next-Token | EEG1 only | 0 | 2580322 | 0:06:45 | 1 | 0.113 | 63 | 53 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG1 only | 1 | 2580977 | 0:08:06 | 1 | 0.135 | 72 | 62 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG1 only | 2 | 2580978 | 0:06:15 | 1 | 0.104 | 58 | 48 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG1 only | 3 | 2580979 | 0:06:06 | 1 | 0.102 | 52 | 42 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG1 only | 4 | 2580980 | 0:08:09 | 1 | 0.136 | 67 | 57 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG2 only | 0 | 2580981 | 0:07:54 | 1 | 0.132 | 84 | 74 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG2 only | 1 | 2580982 | 0:05:31 | 1 | 0.092 | 56 | 46 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG2 only | 2 | 2580983 | 0:06:15 | 1 | 0.104 | 67 | 57 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG2 only | 3 | 2580984 | 0:08:44 | 1 | 0.146 | 78 | 68 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG2 only | 4 | 2580985 | 0:05:39 | 1 | 0.094 | 49 | 39 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG3 only | 0 | 2580986 | 0:06:11 | 1 | 0.103 | 62 | 52 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG3 only | 1 | 2580987 | 0:05:02 | 1 | 0.084 | 50 | 40 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG3 only | 2 | 2580988 | 0:03:56 | 1 | 0.066 | 40 | 30 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG3 only | 3 | 2580989 | 0:08:14 | 1 | 0.137 | 80 | 70 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | EEG3 only | 4 | 2580990 | 0:04:02 | 1 | 0.067 | 42 | 32 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | 3-channel, patience 10 | 0 | 2613266 | 0:05:53 | 1 | 0.098 | 58 | 48 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | 3-channel, patience 10 | 1 | 2613267 | 0:06:46 | 1 | 0.113 | 55 | 45 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | 3-channel, patience 10 | 2 | 2613268 | 0:09:05 | 1 | 0.151 | 75 | 65 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | 3-channel, patience 10 | 3 | 2613270 | 0:09:36 | 1 | 0.160 | 97 | 87 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | 3-channel, patience 10 | 4 | 2613271 | 0:10:46 | 1 | 0.179 | 92 | 82 | sacct (hamehdi) + log (archive) |
| Fine-tune | SleepFM Next-Token | 3-channel benchmark | 0 | 1390519 | 0:15:11 | 1 | 0.253 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM Next-Token | 3-channel benchmark | 1 | 1390520 | 0:13:34 | 1 | 0.226 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM Next-Token | 3-channel benchmark | 2 | 1390521 | 0:13:40 | 1 | 0.228 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM Next-Token | 3-channel benchmark | 3 | 1390522 | 0:12:46 | 1 | 0.213 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | SleepFM Next-Token | 3-channel benchmark | 4 | 1390523 | 0:13:07 | 1 | 0.219 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | BIOT | EEG1 only | 0 | 2580385 | 6:42:08 | 1 | 6.702 | 15 | 5 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG1 only | 1 | 2613241 | 12:28:31 | 1 | 12.475 | 22 | 12 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG1 only | 2 | 2613242 | 8:56:41 | 1 | 8.945 | 17 | 7 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG1 only | 3 | 2613243 | 9:04:38 | 1 | 9.077 | 16 | 6 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG1 only | 4 | 2613244 | 16:47:39 | 1 | 16.794 | 30 | 20 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG2 only | 0 | 2613245 | 13:09:45 | 1 | 13.162 | 23 | 13 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG2 only | 1 | 2613246 | 12:36:12 | 1 | 12.603 | 23 | 13 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG2 only | 2 | 2613247 | 11:34:20 | 1 | 11.572 | 30 | 20 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG2 only | 3 | 2613248 | 9:47:37 | 1 | 9.794 | 26 | 16 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG2 only | 4 | 2613249 | 9:00:38 | 1 | 9.011 | 16 | 6 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG3 only | 0 | 2613250 | 9:02:14 | 1 | 9.037 | 23 | 13 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG3 only | 1 | 2613251 | 10:46:52 | 1 | 10.781 | 24 | 14 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG3 only | 2 | 2613252 | 9:45:33 | 1 | 9.759 | 17 | 7 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG3 only | 3 | 2613253 | 8:44:08 | 1 | 8.736 | 22 | 12 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | EEG3 only | 4 | 2613254 | 10:55:56 | 1 | 10.932 | 28 | 18 | sacct (hamehdi) + log (archive) |
| Fine-tune | BIOT | 3-channel benchmark | 0 | 539039,561752 | 20:03:15 | 1 | 20.054 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | BIOT | 3-channel benchmark | 1 | 539040 | 11:55:30 | 1 | 11.925 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | BIOT | 3-channel benchmark | 2 | 539041,561779 | 15:45:49 | 1 | 15.764 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | BIOT | 3-channel benchmark | 3 | 539042 | 10:54:59 | 1 | 10.916 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |
| Fine-tune | BIOT | 3-channel benchmark | 4 | 539043,561780 | 14:15:26 | 1 | 14.257 | n/a | n/a | sacct (hamehdi); epoch count not recovered -- see recovery command |

Total across every row above: **275.6 GPU-hours** (82 job-rows, 85 distinct SLURM jobs). This is historical training cost for the runs already in `results/`, not part of this task's own GPU-hours (see "GPU-hours used" at the end of this file, which covers only the jobs run while producing this report).

**Not recovered:** epoch counts for the SleepFM and BIOT **3-channel benchmark** fine-tuning runs (10 + 8 job-rows above, elapsed/GPU-hours still confirmed via sacct) -- their `.log` files were written under the old account's home directory (`/users/hamehdi/...`, mode 0700, not group-readable) and predate the Nov-2026 archive. Exact command to recover them, **run from the hamehdi account**, writing to a location `hamehd` can read:

```
mkdir -p /scratch/project_2019517/archive/recovered_logs_07_e5 && cd /users/hamehdi/projects/sleepfm-mesa/logs && cp ft_ss_fullcohort_ft_ss_fullcohort_1418927.log ft_ss_fullcohort_ft_ss_fullcohort_1418928.log ft_ss_fullcohort_ft_ss_fullcohort_1418929.log ft_ss_fullcohort_ft_ss_fullcohort_1418930.log ft_ss_fullcohort_ft_ss_fullcohort_1418931.log ft_ss_fullcohort_ft_ss_fullcohort_1390519.log ft_ss_fullcohort_ft_ss_fullcohort_1390520.log ft_ss_fullcohort_ft_ss_fullcohort_1390521.log ft_ss_fullcohort_ft_ss_fullcohort_1390522.log ft_ss_fullcohort_ft_ss_fullcohort_1390523.log biot_fullcohort_biot_fullcohort_539039.log biot_fullcohort_biot_fullcohort_539040.log biot_fullcohort_biot_fullcohort_539041.log biot_fullcohort_biot_fullcohort_539042.log biot_fullcohort_biot_fullcohort_539043.log biot_resume_biot_resume_561752.log biot_resume_biot_resume_561779.log biot_resume_biot_resume_561780.log /scratch/project_2019517/archive/recovered_logs_07_e5/ && chmod -R g+r /scratch/project_2019517/archive/recovered_logs_07_e5
```

### E6: accuracy against compute
Fixed from the first version of this report: each condition now shows its OWN latency and FLOPs (previously both 1-channel and 3-channel rows repeated the same 3-channel-only figures).


*Compared: models, down each column within each block, higher is better.*
<!-- spec: block=Condition; items=models; value_cols=Pooled macro F1 (95% CI); bold=down; green=none -->
| Model | Condition | Pooled macro F1 (95% CI) | Total params | FLOPs / epoch | GPU batch=128 median latency (ms, mean of 3 repeats, exclusive-node) |
|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG3 only | 0.6427 [0.6373, 0.6481] | 7,019,179 | 9,796,608 | 3.036 |
| SleepFM From-Scratch | 3-channel benchmark | 0.6984 [0.6938, 0.7031] | 7,019,179 | 29,389,824 | 3.514 |
| SleepFM Next-Token | EEG3 only | **0.6584 [0.6528, 0.6642]** | 7,019,179 | 9,796,608 | 3.036 |
| SleepFM Next-Token | 3-channel benchmark | 0.6960 [0.6914, 0.7009] | 7,019,179 | 29,389,824 | 3.514 |
| BIOT | EEG3 only | 0.6538 [0.6479, 0.6596] | 3,443,991 | 381,982,720 | 4.036 |
| BIOT | 3-channel benchmark | **0.7084 [0.7036, 0.7130]** | 3,443,991 | 1,145,943,040 | 7.249 |

"1 channel" uses EEG3 (the C4-M1 electrode match to MESA's own lead) as the representative single channel; "3 channels" uses each model's 3-channel benchmark run. FLOPs and latency here are each model's end-to-end/encoder-only cost at that condition's actual channel count (BIOT: end-to-end; SleepFM: encoder only, since the head's cost is amortized over a whole night rather than per epoch -- see E2).

## GPU-hours used
Covers every job run while producing this report (Phase 1 and this follow-up), via `sacct -u hamehd` (excluding `.batch`/`.extern` sub-steps). Part A, C, E1, and E4 needed no GPU and ran on a login-node CPU venv, so they do not appear here. This is this task's OWN compute cost, distinct from Part E5's historical training-cost ledger for the runs already in `results/`.

*No marks: job list, GPU-hours summed below.*
<!-- spec: items=job list; marks=none; note=job list, GPU-hours summed below -->
| Job ID | Purpose | Elapsed | GPUs | State |
|---|---|---|---|---|
| 3084148 | Phase 1: BIOT eval rerun (first attempt, pre-incident) | 01:15:05 | 1 | completed |
| 3092867 | Phase 1: BIOT eval rerun (clean, verified, --no_checkpoint_link) | 01:10:45 | 1 | completed |
| 3093075 | Phase 1: Part E2 FLOPs (first attempt) | 00:00:10 | 1 | completed |
| 3093089 | Phase 1: Part E3 exclusive-node probe (returned in ~3s) | 00:00:03 | 4 | completed |
| 3093108 | Phase 1: Part E3 shared-node (first attempt, superseded this round) | 00:00:07 | 1 | completed |
| 3093150 | Phase 1: Part E3 shared-node (superseded this round) | 00:01:15 | 1 | completed |
| 3093319 | Phase 1: Part E2 FLOPs (canonical run, still used above) | 00:00:10 | 1 | completed |
| 3094289 | Phase 1: Part E3 exclusive-node (superseded this round) | 00:01:17 | 4 | completed |
| 3105302 | Follow-up: E3 shared-node v2, attempt 1 -- crashed, missing `module load python-pytorch/2.10` | 00:00:02 | 1 | FAILED |
| 3105303 | Follow-up: E3 exclusive-GPU v2, attempt 1 -- same missing-module crash | 00:00:02 | 4 | FAILED |
| 3105315 | Follow-up: E3 shared-node v2, attempt 2 -- ran CPU-only (CUDA silently unavailable, same missing module), cancelled once diagnosed | 00:06:11 | 1 | cancelled |
| 3105316 | Follow-up: E3 exclusive-GPU v2, attempt 2 -- completed but produced 0 rows (same missing module) | 00:00:10 | 4 | completed, unused |
| 3105366 | Follow-up: E3 exclusive-GPU v2, attempt 3 -- cancelled while still queued, once the fix was ready | 00:00:00 | 0 | cancelled |
| 3105306 | Follow-up: E3 exclusive-CPU v2 (real benchmark, used above) | 00:18:00 | 0 | completed |
| 3105424 | Follow-up: E3 shared-node v2 (real benchmark, used above) | 00:23:45 | 1 | completed |
| 3105425 | Follow-up: E3 exclusive-GPU v2 (real benchmark, used above) | 00:00:46 | 4 | completed |

Total across every row above: **3.11 GPU-hours** (includes the three failed/superseded attempts caused by a missing `module load python-pytorch/2.10` line in the new v2 SLURM scripts -- caught quickly, since the shared-node job's log showed the same CUDA driver-mismatch warning the exclusive-node job had silently eaten; fixed by copying the module-load line from the original, working Phase 1 scripts, which I had omitted when rewriting the benchmark for item 3).

