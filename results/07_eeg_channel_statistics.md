# EEG Channel Statistics and Compute Analysis (Phase 1)

**Finding:** A single EEG channel scores 0.03 to 0.07 macro F1 below its model's 3-channel run, and none is equivalent within 0.02. Between channels, differences are detectable but small: at the 0.01 margin, all 3 of SleepFM From-Scratch's channel pairs are TOST-equivalent, 2 of 3 of SleepFM Next-Token's are, and 1 of 3 of BIOT's is; at the 0.02 margin, all 9 of 9 channel pairs across the three models are equivalent. BIOT's EEG2 is about 0.01 lower than its other two channels. Fz-Cz has the lowest Wake recall and the highest N2 recall in all three models.

(Numbers are the new PRIMARY statistic, epoch-pooled macro F1 -- see D.1/D.3. Verified directly, including a recount from the raw per-pair verdicts (task item 2 of this round): single-channel-vs-reference diffs range 0.0331-0.0654 -> "0.03 to 0.07" [D.2]; all 9 such pairs are "not equivalent" at margin 0.02 [D.3]; among the 3 channel-pairs per model, margin 0.01 gives From-Scratch 3/3 equivalent, Next-Token 2/3 (the third is "inconclusive", not "equivalent" -- EEG1 vs EEG3, 90% CI [0.0005, 0.0101], just crosses the margin), BIOT 1/3 (EEG1 vs EEG3 only; the other two are "inconclusive"); margin 0.02 gives 3/3 for every model, 9/9 overall -- this recount matches the counts expected from the underlying 90% intervals exactly, no discrepancy found. BIOT EEG1 vs EEG2 diff=+0.0092, EEG2 vs EEG3 diff=-0.0112, both -> "about 0.01 lower" for EEG2 [D.2]; pooled Wake/N2 recall confirms EEG1/Fz-Cz is lowest-Wake and highest-N2 in all three models [D.5].)

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
| Epoch-pooled macro F1 (PRIMARY statistic) | Pool every valid 30s epoch from the subjects in scope into one 5x5 confusion matrix, take macro F1 from that pooled matrix's counts. Computed per run (all 2056 subjects) and per fold; see D.1 |
| Mean per-subject macro F1 (SECONDARY statistic) | The mean of each subject's OWN, separately-computed macro F1. This file's statistic before this revision; kept as a secondary, clearly-labeled statistic -- see D.1's note on why the two differ |
| Cluster bootstrap | Resampling subjects (not epochs) with replacement, 2000 resamples, recomputing the statistic (pooled-count macro F1 for the primary stat, or the mean for the secondary stat) each draw, for a CI |
| Hodges-Lehmann estimate | The median of all pairwise averages of the paired differences -- the matched estimator for a Wilcoxon signed-rank test; estimates a shift in the TYPICAL (median) subject, not the mean |
| Rank-biserial | Effect size for the paired Wilcoxon signed-rank test, in [-1, 1]; sign follows condition A minus condition B |
| Holm adjustment | Step-down p-value correction across the pairs tested within one model/stage, controlling the family-wise error rate |
| TOST | Two one-sided test for equivalence: a result is "equivalent" at a margin when the 90% CI of the paired difference lies entirely inside the margin |
| 5-minute chunk | The SleepFM encoder's real unit of inference: one forward call processes 5 minutes (38400 samples @ 128Hz) of signal, not a 30s epoch -- see E2 |
| FLOPs | Multiply-accumulate-dominated floating point operations; this file reports two independent counting methods (FlopCounterMode and fvcore, plus hand-added terms neither covers) -- see Part E2 for exactly what each does and doesn't count, and why they disagree |
| Shared-node | A benchmark run on the normal shared GPU partition, alongside other users' jobs on the same physical node |
| Exclusive-node | A benchmark run with the whole node reserved, no other job running on it |

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
| Part D.1 PRIMARY: per-subject confusion-matrix counts' subject set matches the per-subject table, all 14 groups | OK |
| Part D.1 PRIMARY: run-pooled macro F1's per-fold values reproduce the documented 3-channel-benchmark fold means (02/04/06) exactly | OK, diff=0.0000 on all 3 |
| Part D.5: pretraining-subject-overlap check (D.6 note) | confirmed: all 2056 fine-tuning/test subjects are in the SleepFM pretraining split (self-supervised, no labels used) |
| Part E2: verified real SleepFM encoder input shape against sleepfm/models/dataset.py + pretraining config.json | 5-minute chunk (38400 samples), not a 30s epoch -- corrected |
| Part E2: second independent FLOPs method (fvcore) + hand-added LSTM/SDPA-attention/STFT/elementwise terms | done -- explains the gap vs. FlopCounterMode, see note |
| Part E3: redone on the corrected encoder shape, with lscpu/nvidia-smi hardware capture per run | done |
| Part E5: training cost from sacct | recovered, 275.6 GPU-hours across 82 rows -- see note below |
| `scripts/results_tables.py --check` | 0 issues (run after --fix, see Part F commit) |
| `scripts/check_results_numbers.py` | run after --fix; see commit |
| Unit tests (test_channel_stats.py, test_compute_profile.py, test_experiment_paths.py, full suite) | see commit |
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
Saved to `~/analysis/eeg_channels/per_subject_metrics.csv` (28,784 rows: 14 (model, condition) groups x 2056 subjects, not in git): one row per (model, condition, fold, subject_id), with `n_valid_epochs`, `accuracy`, `kappa`, `macro_f1`, and `f1_<Stage>`/`recall_<Stage>` for Wake/N1/N2/N3/REM, computed from the mask (valid epochs only) and over only the classes present in that subject -- this is the SECONDARY statistic's source table. A second file, `fold_level_metrics.csv` (70 rows), holds the epoch-flattened per-fold macro F1/accuracy used by D.1 and the reproduction check above. A third file, `per_subject_confusion.npz` (built by `scripts/analysis/build_confusion.py` from the same raw predictions), holds each subject's own 5x5 confusion matrix per (model, condition) -- the source for the PRIMARY (epoch-pooled) statistic's subject-level cluster bootstrap: summing confusion matrices over any subset of subjects is exact, so resampling subjects and re-pooling counts each draw (D.1-D.3) is exact, unlike resampling already-averaged per-subject macro F1 values.


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

`kappa` is undefined when a subject's targets AND predictions are both constant (never happened here). `f1_<Stage>`/`recall_<Stage>` are undefined when that stage never occurs in a subject's valid epochs for that night. Each undefined-row count divides evenly by 14 (the number of (model, condition) groups): 294/14=21, 2394/14=171, 616/14=44 -- the same 21/171/44 subjects lack N1/N3/REM respectively in every condition, because this is a property of their one night's ground-truth labels, not of the model or channel used to predict it. This cross-check held exactly. Note: this undefined-value issue is specific to the per-subject SECONDARY statistic -- the PRIMARY (epoch-pooled) statistic never drops a subject for an undefined per-class value, since pooling is at the epoch level (see D.5).

## Part D: statistics

### D.1: epoch-pooled macro F1 (PRIMARY statistic)
Pool every valid epoch from the subjects in scope into one 5x5 confusion matrix, take macro F1 from its pooled counts -- computed per run (all 2056 subjects together, "Run-pooled" below) and per fold ("Per-fold macro F1" below, one pooled matrix per fold -- this is the same methodology already published in results/02/04/06, and the per-fold values below reproduce those exactly). The 95% CI on the run-pooled value comes from a cluster bootstrap: resample subjects with replacement, re-sum their confusion matrices, recompute macro F1 from the resampled pooled counts, 2000 draws.

*Compared: models, down each column within each block, higher is better.*
<!-- spec: block=Condition; items=models; value_cols=Run-pooled macro F1 (95% CI); bold=down; green=none -->
| Condition | Model | Run-pooled macro F1 (95% CI) | Per-fold macro F1 (folds 0-4) | n subjects |
|---|---|---|---|---|
| EEG1 only | SleepFM From-Scratch | 0.6836 [0.6783, 0.6887] | 0.6899, 0.6919, 0.6610, 0.6934, 0.6812 | 2056 |
| EEG2 only | SleepFM From-Scratch | 0.6842 [0.6795, 0.6892] | 0.6798, 0.7025, 0.6757, 0.6872, 0.6761 | 2056 |
| EEG3 only | SleepFM From-Scratch | 0.6887 [0.6837, 0.6935] | 0.6966, 0.6947, 0.6947, 0.6923, 0.6646 | 2056 |
| 3-channel, patience 10 | SleepFM From-Scratch | 0.7307 [0.7270, 0.7345] | 0.7139, 0.7446, 0.7324, 0.7357, 0.7265 | 2056 |
| 3-channel benchmark | SleepFM From-Scratch | 0.7451 [0.7414, 0.7488] | 0.7385, 0.7496, 0.7446, 0.7521, 0.7404 | 2056 |
| EEG1 only | SleepFM Next-Token | **0.7015 [0.6960, 0.7069]** | 0.6960, 0.7136, 0.6978, 0.7002, 0.6996 | 2056 |
| EEG2 only | SleepFM Next-Token | **0.6964 [0.6910, 0.7018]** | 0.6963, 0.7036, 0.6935, 0.7092, 0.6804 | 2056 |
| EEG3 only | SleepFM Next-Token | 0.6963 [0.6913, 0.7014] | 0.6981, 0.7051, 0.6905, 0.7060, 0.6821 | 2056 |
| 3-channel, patience 10 | SleepFM Next-Token | **0.7346 [0.7308, 0.7385]** | 0.7270, 0.7381, 0.7373, 0.7408, 0.7297 | 2056 |
| 3-channel benchmark | SleepFM Next-Token | 0.7395 [0.7356, 0.7432] | 0.7350, 0.7461, 0.7373, 0.7449, 0.7338 | 2056 |
| EEG1 only | BIOT | 0.6976 [0.6926, 0.7028] | 0.6922, 0.7074, 0.6952, 0.6951, 0.6982 | 2056 |
| EEG2 only | BIOT | 0.6885 [0.6831, 0.6935] | 0.6840, 0.7043, 0.6779, 0.6900, 0.6872 | 2056 |
| EEG3 only | BIOT | **0.6997 [0.6946, 0.7045]** | 0.6937, 0.7164, 0.6980, 0.6996, 0.6915 | 2056 |
| 3-channel benchmark | BIOT | **0.7539 [0.7500, 0.7576]** | 0.7551, 0.7626, 0.7489, 0.7508, 0.7534 | 2056 |

**Run-pooled vs. per-fold (both epoch-pooled, should be close):** the run-pooled value differs from the mean of the 5 per-fold values by well under 0.001 for every row here (e.g. SleepFM From-Scratch / EEG1 only: run-pooled=0.6836, mean of per-fold=0.68348, diff=0.0001) -- expected, since pooling all 2056 subjects at once is nearly the same operation as pooling each fold's ~411 subjects and then averaging 5 nearly-equal-size, nearly-equal-class-balance pools; the small residual gap is exactly the same mild nonlinearity (macro F1 is a nonlinear function of pooled counts) that produces the much larger gap against the SECONDARY statistic below.

**SECONDARY statistic, for comparison: mean per-subject macro F1 (this file's statistic before this revision) vs. the fold mean:**
This is the mean of each subject's OWN, separately-computed macro F1, bootstrapped over subjects -- a genuinely different statistic from the epoch-pooled one above, not just a different estimate of the same quantity. Kept here because D.2/D.4's Wilcoxon/Friedman tests (necessarily per-subject) are secondary analyses of this statistic, not the primary one.

*Compared: models, down each column within each block, higher is better.*
<!-- spec: block=Condition; items=models; value_cols=Mean per-subject macro F1 (95% CI),Fold mean +/- std (epoch-flattened); bold=down; green=none -->
| Condition | Model | Mean per-subject macro F1 (95% CI) | Fold mean +/- std (epoch-flattened) | n subjects |
|---|---|---|---|---|
| 3-channel benchmark | BIOT | **0.7084 [0.7036, 0.7130]** | **0.7541 +/- 0.0053** | 2056 |
| EEG1 only | BIOT | 0.6498 [0.6443, 0.6554] | 0.6976 +/- 0.0059 | 2056 |
| EEG2 only | BIOT | 0.6415 [0.6356, 0.6473] | 0.6887 +/- 0.0098 | 2056 |
| EEG3 only | BIOT | 0.6538 [0.6479, 0.6596] | **0.6998 +/- 0.0098** | 2056 |
| 3-channel benchmark | SleepFM From-Scratch | 0.6984 [0.6938, 0.7031] | 0.7451 +/- 0.0058 | 2056 |
| 3-channel, patience 10 | SleepFM From-Scratch | 0.6862 [0.6817, 0.6909] | 0.7306 +/- 0.0114 | 2056 |
| EEG1 only | SleepFM From-Scratch | 0.6415 [0.6363, 0.6469] | 0.6835 +/- 0.0134 | 2056 |
| EEG2 only | SleepFM From-Scratch | 0.6427 [0.6371, 0.6486] | 0.6843 +/- 0.0112 | 2056 |
| EEG3 only | SleepFM From-Scratch | 0.6427 [0.6373, 0.6481] | 0.6886 +/- 0.0135 | 2056 |
| 3-channel benchmark | SleepFM Next-Token | 0.6960 [0.6914, 0.7009] | 0.7394 +/- 0.0057 | 2056 |
| 3-channel, patience 10 | SleepFM Next-Token | **0.6929 [0.6885, 0.6974]** | **0.7346 +/- 0.0059** | 2056 |
| EEG1 only | SleepFM Next-Token | **0.6625 [0.6568, 0.6680]** | **0.7014 +/- 0.0070** | 2056 |
| EEG2 only | SleepFM Next-Token | **0.6596 [0.6538, 0.6656]** | **0.6966 +/- 0.0109** | 2056 |
| EEG3 only | SleepFM Next-Token | **0.6584 [0.6528, 0.6642]** | 0.6964 +/- 0.0101 | 2056 |

**Why mean-per-subject and fold-mean differ (consistently, by about 0.04-0.05 here):** macro F1 is a nonlinear (harmonic-mean-based) function of per-class counts, and several classes -- especially N1, and N3/REM for some subjects -- occur only a handful of times in a single subject's single night. A few misclassified epochs in a class with few occurrences swings that subject's F1 for that class drastically; across 2056 subjects this does not average out the way linear statistics would, producing a systematic downward bias in the subject-averaged statistic relative to the epoch-pooled one (both the PRIMARY run-pooled statistic above AND the fold-mean statistic are epoch-pooled, so they agree closely with each other and both disagree with this per-subject mean in the same direction, by about the same amount).

### D.2: paired differences within each model
**PRIMARY statistic (epoch-pooled):** no Wilcoxon p-value here -- the pooled statistic is one number per run, not a per-subject sample, so there is nothing to rank-sum. The paired bootstrap CI (resample subjects, same resampled indices for both conditions, re-pool counts, recompute the macro F1 difference) is the significance signal: does the 95% CI exclude zero.

*No marks: paired comparison statistics, not a ranked comparison.*
<!-- spec: items=paired comparisons (primary, epoch-pooled); marks=none; note=paired comparison statistics, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | Pooled macro F1 diff (A-B) | Bootstrap 95% CI | 95% CI | n subjects |
|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | -0.0007 | [-0.0049, 0.0036] | includes zero | 2056 |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | -0.0052 | [-0.0104, 0.0004] | includes zero | 2056 |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | -0.0045 | [-0.0094, 0.0012] | includes zero | 2056 |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0471 | [-0.0511, -0.0433] | excludes zero | 2056 |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0464 | [-0.0499, -0.0428] | excludes zero | 2056 |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0419 | [-0.0458, -0.0382] | excludes zero | 2056 |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 0.0051 | [0.0005, 0.0096] | excludes zero | 2056 |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.0052 | [-0.0005, 0.0111] | includes zero | 2056 |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 0.0001 | [-0.0055, 0.0062] | includes zero | 2056 |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0331 | [-0.0370, -0.0293] | excludes zero | 2056 |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0382 | [-0.0421, -0.0344] | excludes zero | 2056 |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0383 | [-0.0425, -0.0347] | excludes zero | 2056 |
| BIOT | EEG1 only | EEG2 only |  | 0.0092 | [0.0050, 0.0133] | excludes zero | 2056 |
| BIOT | EEG1 only | EEG3 only |  | -0.0021 | [-0.0071, 0.0030] | includes zero | 2056 |
| BIOT | EEG2 only | EEG3 only |  | -0.0112 | [-0.0165, -0.0056] | excludes zero | 2056 |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0562 | [-0.0594, -0.0530] | excludes zero | 2056 |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0654 | [-0.0691, -0.0617] | excludes zero | 2056 |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0542 | [-0.0581, -0.0507] | excludes zero | 2056 |

Diff = condition A minus condition B.

**SECONDARY statistic (mean per-subject macro F1), with Wilcoxon, Holm, and the Hodges-Lehmann estimate added:** Wilcoxon/Holm test and the Hodges-Lehmann estimate target a shift in the TYPICAL (median) subject; the bootstrap CI targets a shift in the MEAN. These can legitimately disagree under a skewed per-subject diff distribution.

*No marks: paired comparison statistics, not a ranked comparison.*
<!-- spec: items=paired comparisons (secondary, mean per-subject); marks=none; note=paired comparison statistics, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | Mean diff macro F1 | Median diff macro F1 | Hodges-Lehmann diff macro F1 | Median diff accuracy | Median diff kappa | Wilcoxon p | Rank-biserial | Bootstrap 95% CI (mean macro F1 diff) | Holm p | n subjects |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | -0.0012 | -0.0051 | -0.0050 | 0.0009 | 0.0005 | 0.0001 | -0.099 | [-0.0052, 0.0031] | 0.0003 | 2056 |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | -0.0012 | 0.0022 | 0.0012 | 0.0049 | 0.0067 | 0.4594 | 0.019 | [-0.0063, 0.0042] | 0.4594 | 2056 |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | -0.0000 | 0.0058 | 0.0041 | 0.0034 | 0.0057 | 0.0099 | 0.066 | [-0.0049, 0.0056] | 0.0198 | 2056 |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0447 | -0.0333 | -0.0359 | -0.0154 | -0.0240 | 5.19e-142 | -0.646 | [-0.0484, -0.0412] | 3.12e-141 | 2056 |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0435 | -0.0283 | -0.0316 | -0.0168 | -0.0258 | 3.65e-137 | -0.635 | [-0.0472, -0.0400] | 1.83e-136 | 2056 |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0435 | -0.0319 | -0.0349 | -0.0181 | -0.0283 | 8.62e-133 | -0.624 | [-0.0472, -0.0399] | 3.45e-132 | 2056 |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 0.0029 | -0.0029 | -0.0023 | 0.0000 | -0.0010 | 0.0480 | -0.050 | [-0.0013, 0.0073] | 0.0480 | 2056 |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.0041 | 0.0052 | 0.0041 | 0.0008 | 0.0017 | 0.0037 | 0.074 | [-0.0014, 0.0097] | 0.0074 | 2056 |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 0.0011 | 0.0076 | 0.0059 | 0.0000 | 0.0019 | 1.38e-05 | 0.111 | [-0.0044, 0.0070] | 4.15e-05 | 2056 |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0304 | -0.0194 | -0.0215 | -0.0131 | -0.0191 | 6.35e-84 | -0.494 | [-0.0341, -0.0268] | 3.17e-83 | 2056 |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0333 | -0.0177 | -0.0205 | -0.0128 | -0.0180 | 9.05e-83 | -0.491 | [-0.0369, -0.0296] | 3.62e-82 | 2056 |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0345 | -0.0229 | -0.0254 | -0.0130 | -0.0188 | 2.92e-93 | -0.522 | [-0.0383, -0.0307] | 1.75e-92 | 2056 |
| BIOT | EEG1 only | EEG2 only |  | 0.0083 | 0.0051 | 0.0043 | 0.0000 | -0.0001 | 0.0010 | 0.084 | [0.0040, 0.0126] | 0.0020 | 2056 |
| BIOT | EEG1 only | EEG3 only |  | -0.0041 | -0.0006 | -0.0024 | 0.0000 | 0.0004 | 0.1587 | -0.036 | [-0.0096, 0.0014] | 0.1587 | 2056 |
| BIOT | EEG2 only | EEG3 only |  | -0.0123 | -0.0071 | -0.0081 | 0.0007 | -0.0001 | 3.97e-07 | -0.129 | [-0.0179, -0.0063] | 1.19e-06 | 2056 |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0586 | -0.0423 | -0.0475 | -0.0333 | -0.0466 | 1.04e-234 | -0.833 | [-0.0625, -0.0549] | 5.22e-234 | 2056 |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0669 | -0.0489 | -0.0530 | -0.0325 | -0.0468 | 7.50e-244 | -0.849 | [-0.0709, -0.0630] | 4.50e-243 | 2056 |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0546 | -0.0395 | -0.0433 | -0.0322 | -0.0450 | 3.08e-201 | -0.771 | [-0.0586, -0.0508] | 1.23e-200 | 2056 |

Diff = condition A minus condition B. Holm adjustment is applied within each model across its 6 pairs (EEG1-EEG2, EEG1-EEG3, EEG2-EEG3, and each of the three channels vs. that model's reference).

**Pairs where Holm p < 0.05 but the 95% CI of the MEAN difference includes zero (5 of 18):** the rank-based test calls a shift in the typical subject significant, but the mean-based CI doesn't rule out a zero mean shift -- consistent with a skewed per-subject diff distribution (most subjects shift one way, a few shift hard the other way, pulling the mean's CI toward zero while the sign-dominated rank test stays significant).

*No marks: diagnostic list, not a ranked comparison.*
<!-- spec: items=Holm-significant but mean-CI-includes-zero pairs; marks=none; note=diagnostic list, not a ranked comparison -->
| Model | Condition A | Condition B | Holm p | Mean diff | Bootstrap 95% CI (mean) | Hodges-Lehmann diff |
|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only | 0.0003 | -0.0012 | [-0.0052, 0.0031] | -0.0050 |
| SleepFM From-Scratch | EEG2 only | EEG3 only | 0.0198 | -0.0000 | [-0.0049, 0.0056] | 0.0041 |
| SleepFM Next-Token | EEG1 only | EEG2 only | 0.0480 | 0.0029 | [-0.0013, 0.0073] | -0.0023 |
| SleepFM Next-Token | EEG1 only | EEG3 only | 0.0074 | 0.0041 | [-0.0014, 0.0097] | 0.0041 |
| SleepFM Next-Token | EEG2 only | EEG3 only | 4.15e-05 | 0.0011 | [-0.0044, 0.0070] | 0.0059 |


### D.2 extended: paired t-test against the paired Wilcoxon (task item 1)
Per-subject macro F1 and accuracy, same 2056 subjects, same D.2 pairs. The paired t-test (mean difference, sd of differences, t, df, two-sided p, Cohen's dz, analytic 95% CI of the mean difference) is reported next to the paired Wilcoxon (p, rank-biserial, Hodges-Lehmann estimate). Holm adjustment is done SEPARATELY for the t-test p-values and the Wilcoxon p-values, each within one model across its 6 pairs -- these are two different hypotheses about the same data, not one family, so their multiplicity corrections must not interact. Skewness and excess kurtosis of the paired differences are also given (both 0 for a normal distribution).

**Per-subject macro F1:**

*No marks: paired comparison statistics, not a ranked comparison.*
<!-- spec: items=paired t-test vs Wilcoxon (macro F1); marks=none; note=paired comparison statistics, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | Mean diff | sd of diff | t | df | t-test p | t-test Holm p | Cohen's dz | t-test 95% CI | Wilcoxon p | Wilcoxon Holm p | Rank-biserial | Hodges-Lehmann diff | Skewness | Excess kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | -0.0012 | 0.0993 | -0.552 | 2055 | 0.5809 | 1.0000 | -0.012 | [-0.0055, 0.0031] | 0.0001 | 0.0003 | -0.099 | -0.0050 | +0.628 | +8.318 |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | -0.0012 | 0.1160 | -0.484 | 2055 | 0.6284 | 1.0000 | -0.011 | [-0.0063, 0.0038] | 0.4594 | 0.4594 | +0.019 | +0.0012 | -0.107 | +4.479 |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | -0.0000 | 0.1211 | -0.011 | 2055 | 0.9912 | 1.0000 | -0.000 | [-0.0053, 0.0052] | 0.0099 | 0.0198 | +0.066 | +0.0041 | -0.319 | +4.689 |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0447 | 0.0823 | -24.628 | 2055 | 1.43e-117 | 8.57e-117 | -0.543 | [-0.0483, -0.0411] | 5.19e-142 | 3.12e-141 | -0.646 | -0.0359 | -1.330 | +6.430 |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0435 | 0.0848 | -23.257 | 2055 | 2.11e-106 | 8.44e-106 | -0.513 | [-0.0472, -0.0398] | 3.65e-137 | 1.83e-136 | -0.635 | -0.0316 | -1.928 | +6.477 |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0435 | 0.0840 | -23.465 | 2055 | 4.46e-108 | 2.23e-107 | -0.518 | [-0.0471, -0.0398] | 8.62e-133 | 3.45e-132 | -0.624 | -0.0349 | -1.627 | +5.992 |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 0.0029 | 0.1045 | 1.266 | 2055 | 0.2055 | 0.4110 | +0.028 | [-0.0016, 0.0074] | 0.0480 | 0.0480 | -0.050 | -0.0023 | +0.434 | +11.936 |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.0041 | 0.1226 | 1.505 | 2055 | 0.1325 | 0.3975 | +0.033 | [-0.0012, 0.0094] | 0.0037 | 0.0074 | +0.074 | +0.0041 | -0.001 | +7.280 |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 0.0011 | 0.1269 | 0.411 | 2055 | 0.6813 | 0.6813 | +0.009 | [-0.0043, 0.0066] | 1.38e-05 | 4.15e-05 | +0.111 | +0.0059 | -0.155 | +6.732 |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0304 | 0.0831 | -16.578 | 2055 | 4.99e-58 | 2.00e-57 | -0.366 | [-0.0340, -0.0268] | 6.35e-84 | 3.17e-83 | -0.494 | -0.0215 | -2.327 | +12.770 |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0333 | 0.0875 | -17.254 | 2055 | 2.11e-62 | 1.06e-61 | -0.381 | [-0.0371, -0.0295] | 9.05e-83 | 3.62e-82 | -0.491 | -0.0205 | -2.207 | +8.815 |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0345 | 0.0877 | -17.821 | 2055 | 3.55e-66 | 2.13e-65 | -0.393 | [-0.0383, -0.0307] | 2.92e-93 | 1.75e-92 | -0.522 | -0.0254 | -2.034 | +9.914 |
| BIOT | EEG1 only | EEG2 only |  | 0.0083 | 0.1026 | 3.649 | 2055 | 0.0003 | 0.0005 | +0.080 | [0.0038, 0.0127] | 0.0010 | 0.0020 | +0.084 | +0.0043 | +0.695 | +10.744 |
| BIOT | EEG1 only | EEG3 only |  | -0.0041 | 0.1259 | -1.459 | 2055 | 0.1446 | 0.1446 | -0.032 | [-0.0095, 0.0014] | 0.1587 | 0.1587 | -0.036 | -0.0024 | -0.157 | +5.233 |
| BIOT | EEG2 only | EEG3 only |  | -0.0123 | 0.1299 | -4.297 | 2055 | 1.81e-05 | 5.44e-05 | -0.095 | [-0.0179, -0.0067] | 3.97e-07 | 1.19e-06 | -0.129 | -0.0081 | -0.295 | +5.023 |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0586 | 0.0820 | -32.407 | 2055 | 1.87e-186 | 9.34e-186 | -0.715 | [-0.0622, -0.0551] | 1.04e-234 | 5.22e-234 | -0.833 | -0.0475 | -2.161 | +8.502 |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0669 | 0.0916 | -33.124 | 2055 | 3.67e-193 | 2.20e-192 | -0.731 | [-0.0708, -0.0629] | 7.50e-244 | 4.50e-243 | -0.849 | -0.0530 | -2.210 | +7.553 |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0546 | 0.0879 | -28.149 | 2055 | 9.83e-148 | 3.93e-147 | -0.621 | [-0.0584, -0.0508] | 3.08e-201 | 1.23e-200 | -0.771 | -0.0433 | -2.021 | +8.359 |

**Per-subject accuracy:**

*No marks: paired comparison statistics, not a ranked comparison.*
<!-- spec: items=paired t-test vs Wilcoxon (accuracy); marks=none; note=paired comparison statistics, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | Mean diff | sd of diff | t | df | t-test p | t-test Holm p | Cohen's dz | t-test 95% CI | Wilcoxon p | Wilcoxon Holm p | Rank-biserial | Hodges-Lehmann diff | Skewness | Excess kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | 0.0030 | 0.0870 | 1.542 | 2055 | 0.1232 | 0.3695 | +0.034 | [-0.0008, 0.0067] | 0.1659 | 0.1659 | +0.036 | +0.0013 | -0.104 | +10.649 |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 0.0013 | 0.1036 | 0.552 | 2055 | 0.5810 | 0.9168 | +0.012 | [-0.0032, 0.0057] | 0.0008 | 0.0025 | +0.085 | +0.0041 | -0.253 | +6.709 |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | -0.0017 | 0.1038 | -0.742 | 2055 | 0.4584 | 0.9168 | -0.016 | [-0.0062, 0.0028] | 0.0566 | 0.1133 | +0.049 | +0.0022 | -0.280 | +6.151 |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0307 | 0.0769 | -18.112 | 2055 | 3.83e-68 | 1.53e-67 | -0.399 | [-0.0340, -0.0274] | 9.19e-87 | 3.68e-86 | -0.506 | -0.0188 | -2.358 | +11.400 |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0337 | 0.0745 | -20.485 | 2055 | 5.16e-85 | 3.10e-84 | -0.452 | [-0.0369, -0.0304] | 9.05e-113 | 5.43e-112 | -0.578 | -0.0206 | -2.477 | +8.900 |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0320 | 0.0743 | -19.498 | 2055 | 7.98e-78 | 3.99e-77 | -0.430 | [-0.0352, -0.0288] | 6.08e-111 | 3.04e-110 | -0.572 | -0.0216 | -2.484 | +11.603 |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 0.0040 | 0.0964 | 1.871 | 2055 | 0.0615 | 0.1844 | +0.041 | [-0.0002, 0.0081] | 0.9774 | 1.0000 | +0.001 | +0.0000 | +0.027 | +14.808 |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.0000 | 0.1139 | 0.011 | 2055 | 0.9913 | 0.9913 | +0.000 | [-0.0049, 0.0050] | 0.5167 | 1.0000 | +0.017 | +0.0007 | -0.135 | +9.825 |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | -0.0040 | 0.1145 | -1.564 | 2055 | 0.1180 | 0.2360 | -0.034 | [-0.0089, 0.0010] | 0.9876 | 1.0000 | +0.000 | +0.0000 | +0.064 | +7.777 |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | -0.0280 | 0.0807 | -15.728 | 2055 | 1.02e-52 | 5.12e-52 | -0.347 | [-0.0315, -0.0245] | 1.46e-85 | 7.32e-85 | -0.503 | -0.0159 | -3.143 | +17.515 |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | -0.0320 | 0.0823 | -17.606 | 2055 | 9.83e-65 | 5.90e-64 | -0.388 | [-0.0355, -0.0284] | 1.73e-102 | 1.04e-101 | -0.551 | -0.0167 | -2.522 | +10.253 |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | -0.0280 | 0.0827 | -15.356 | 2055 | 1.83e-50 | 7.31e-50 | -0.339 | [-0.0316, -0.0244] | 2.52e-79 | 1.01e-78 | -0.482 | -0.0157 | -2.945 | +15.968 |
| BIOT | EEG1 only | EEG2 only |  | 0.0030 | 0.0904 | 1.500 | 2055 | 0.1337 | 0.2675 | +0.033 | [-0.0009, 0.0069] | 0.9966 | 1.0000 | -0.000 | +0.0000 | +0.567 | +10.710 |
| BIOT | EEG1 only | EEG3 only |  | -0.0017 | 0.1050 | -0.729 | 2055 | 0.4663 | 0.4663 | -0.016 | [-0.0062, 0.0029] | 0.9876 | 1.0000 | +0.000 | +0.0000 | -0.232 | +6.387 |
| BIOT | EEG2 only | EEG3 only |  | -0.0047 | 0.1084 | -1.956 | 2055 | 0.0506 | 0.1517 | -0.043 | [-0.0094, 0.0000] | 0.5774 | 1.0000 | -0.014 | -0.0007 | -0.293 | +5.715 |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0485 | 0.0707 | -31.106 | 2055 | 2.07e-174 | 1.24e-173 | -0.686 | [-0.0516, -0.0454] | 7.28e-238 | 4.37e-237 | -0.841 | -0.0374 | -2.596 | +10.644 |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0515 | 0.0788 | -29.619 | 2055 | 7.29e-161 | 3.64e-160 | -0.653 | [-0.0549, -0.0481] | 6.60e-223 | 3.30e-222 | -0.813 | -0.0379 | -2.342 | +8.349 |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | -0.0468 | 0.0753 | -28.194 | 2055 | 3.95e-148 | 1.58e-147 | -0.622 | [-0.0501, -0.0436] | 8.01e-213 | 3.21e-212 | -0.794 | -0.0361 | -2.404 | +10.511 |

**Pairs where the Holm-adjusted t-test and the Holm-adjusted Wilcoxon disagree about significance at alpha=0.05 (5 of 18 for macro F1, 1 of 18 for accuracy):**

*No marks: diagnostic list, not a ranked comparison.*
<!-- spec: items=t-test vs Wilcoxon disagreements; marks=none; note=diagnostic list, not a ranked comparison -->
| Metric | Model | Condition A | Condition B | t-test Holm p | Wilcoxon Holm p |
|---|---|---|---|---|---|
| macro F1 | SleepFM From-Scratch | EEG1 only | EEG2 only | 1.0000 | 0.0003 |
| macro F1 | SleepFM From-Scratch | EEG2 only | EEG3 only | 1.0000 | 0.0198 |
| macro F1 | SleepFM Next-Token | EEG1 only | EEG2 only | 0.4110 | 0.0480 |
| macro F1 | SleepFM Next-Token | EEG1 only | EEG3 only | 0.3975 | 0.0074 |
| macro F1 | SleepFM Next-Token | EEG2 only | EEG3 only | 0.6813 | 4.15e-05 |
| accuracy | SleepFM From-Scratch | EEG1 only | EEG3 only | 0.9168 | 0.0025 |

Every disagreement listed is the SAME direction: Wilcoxon calls it significant, the t-test does not -- the mirror image never happens here.

**On which test is primary (corrected wording, task item 3):** the primary result in this file is the difference in POOLED macro F1 (D.1-D.3's PRIMARY statistic) with its subject-cluster-bootstrap interval and TOST verdict -- not either per-subject test below. Of the two per-subject (SECONDARY-statistic) tests, the paired t-test tests whether the MEAN of the per-subject macro F1 differences is zero; the paired Wilcoxon tests whether there is a shift in the TYPICAL (median) subject. These are different questions about the same data, and with skewed per-subject differences (see the skewness/excess-kurtosis columns above) they can disagree -- as they do for the 5 pairs below, where the mean is near zero (sometimes even the opposite sign from the median) while the typical subject still shifts:

*No marks: diagnostic detail for the 5 disagreeing pairs, not a ranked comparison.*
<!-- spec: items=disagreeing pairs detail; marks=none; note=diagnostic detail for the 5 disagreeing pairs, not a ranked comparison -->
| Model | Condition A | Condition B | Mean diff | Median diff | Hodges-Lehmann diff | Skewness |
|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only | -0.0012 | -0.0051 | -0.0050 | +0.628 |
| SleepFM From-Scratch | EEG2 only | EEG3 only | -0.0000 | +0.0058 | +0.0041 | -0.319 |
| SleepFM Next-Token | EEG1 only | EEG2 only | +0.0029 | -0.0029 | -0.0023 | +0.434 |
| SleepFM Next-Token | EEG1 only | EEG3 only | +0.0041 | +0.0052 | +0.0041 | -0.001 |
| SleepFM Next-Token | EEG2 only | EEG3 only | +0.0011 | +0.0076 | +0.0059 | -0.155 |

Note the SleepFM Next-Token EEG1-vs-EEG2 row: the mean (+0.0029) and the median/Hodges-Lehmann estimate (-0.0029/-0.0023) don't even agree in SIGN -- exactly the kind of case where "the mean" and "the typical subject" are genuinely different questions, not two estimates of the same number. For the large, clearly nonzero single-channel-vs-reference effects elsewhere in this table, both tests agree at p < 0.0001 regardless of this concern.

### D.2 supplement: fold-level paired t-test (task item 2)
A paired t-test on the 5 fold-level (epoch-pooled) macro F1 values for each D.2 pair -- df=4. **Low power** (n=5) **and not independent observations**: in 5-fold cross-validation every pair of folds' TRAINING sets share 3 of the other 4 folds' subjects (each fold's training set is "everyone else"), so the 5 fold-level values are correlated with each other through shared training data, violating the paired t-test's independence assumption. Reported as a labeled supplement only, never as a substitute for the subject-level tests above.

*No marks: low-power supplementary test, not a ranked comparison.*
<!-- spec: items=fold-level paired t-test; marks=none; note=low-power supplementary test, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | n folds | Mean diff | sd of diff | t | df | p |
|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | 5 | -0.0008 | 0.0111 | -0.156 | 4 | 0.8836 |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 5 | -0.0051 | 0.0182 | -0.624 | 4 | 0.5664 |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | 5 | -0.0043 | 0.0138 | -0.699 | 4 | 0.5233 |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | 5 | -0.0471 | 0.0172 | -6.126 | 4 | 0.0036 |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | 5 | -0.0464 | 0.0086 | -12.023 | 4 | 0.0003 |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | 5 | -0.0420 | 0.0165 | -5.698 | 4 | 0.0047 |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 5 | 0.0049 | 0.0106 | 1.023 | 4 | 0.3640 |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 5 | 0.0051 | 0.0092 | 1.231 | 4 | 0.2858 |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 5 | 0.0002 | 0.0026 | 0.192 | 4 | 0.8572 |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | 5 | -0.0331 | 0.0068 | -10.898 | 4 | 0.0004 |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | 5 | -0.0380 | 0.0082 | -10.387 | 4 | 0.0005 |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | 5 | -0.0382 | 0.0085 | -10.092 | 4 | 0.0005 |
| BIOT | EEG1 only | EEG2 only |  | 5 | 0.0089 | 0.0055 | 3.622 | 4 | 0.0223 |
| BIOT | EEG1 only | EEG3 only |  | 5 | -0.0022 | 0.0057 | -0.864 | 4 | 0.4361 |
| BIOT | EEG2 only | EEG3 only |  | 5 | -0.0112 | 0.0058 | -4.328 | 4 | 0.0124 |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0565 | 0.0036 | -34.721 | 4 | 4.11e-06 |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0655 | 0.0058 | -25.315 | 4 | 1.45e-05 |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0543 | 0.0070 | -17.461 | 4 | 6.32e-05 |

### D.3: TOST equivalence
**PRIMARY statistic (epoch-pooled):**

*No marks: equivalence verdicts, not a ranked comparison.*
<!-- spec: items=TOST verdicts (primary, epoch-pooled); marks=none; note=equivalence verdicts, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | Margin | Point diff | 90% CI (pooled macro F1 diff) | Verdict |
|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | 0.01 | -0.0007 | [-0.0044, 0.0029] | equivalent |
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | 0.02 | -0.0007 | [-0.0044, 0.0029] | equivalent |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 0.01 | -0.0052 | [-0.0096, -0.0005] | equivalent |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 0.02 | -0.0052 | [-0.0096, -0.0005] | equivalent |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | 0.01 | -0.0045 | [-0.0088, 0.0002] | equivalent |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | 0.02 | -0.0045 | [-0.0088, 0.0002] | equivalent |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0471 | [-0.0505, -0.0439] | not equivalent |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0471 | [-0.0505, -0.0439] | not equivalent |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0464 | [-0.0494, -0.0433] | not equivalent |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0464 | [-0.0494, -0.0433] | not equivalent |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0419 | [-0.0452, -0.0388] | not equivalent |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0419 | [-0.0452, -0.0388] | not equivalent |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 0.01 | 0.0051 | [0.0012, 0.0089] | equivalent |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 0.02 | 0.0051 | [0.0012, 0.0089] | equivalent |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.01 | 0.0052 | [0.0005, 0.0101] | inconclusive |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 0.02 | 0.0052 | [0.0005, 0.0101] | equivalent |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 0.01 | 0.0001 | [-0.0045, 0.0050] | equivalent |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 0.02 | 0.0001 | [-0.0045, 0.0050] | equivalent |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0331 | [-0.0364, -0.0299] | not equivalent |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0331 | [-0.0364, -0.0299] | not equivalent |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0382 | [-0.0415, -0.0348] | not equivalent |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0382 | [-0.0415, -0.0348] | not equivalent |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | 0.01 | -0.0383 | [-0.0417, -0.0351] | not equivalent |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | 0.02 | -0.0383 | [-0.0417, -0.0351] | not equivalent |
| BIOT | EEG1 only | EEG2 only |  | 0.01 | 0.0092 | [0.0058, 0.0127] | inconclusive |
| BIOT | EEG1 only | EEG2 only |  | 0.02 | 0.0092 | [0.0058, 0.0127] | equivalent |
| BIOT | EEG1 only | EEG3 only |  | 0.01 | -0.0021 | [-0.0062, 0.0022] | equivalent |
| BIOT | EEG1 only | EEG3 only |  | 0.02 | -0.0021 | [-0.0062, 0.0022] | equivalent |
| BIOT | EEG2 only | EEG3 only |  | 0.01 | -0.0112 | [-0.0156, -0.0066] | inconclusive |
| BIOT | EEG2 only | EEG3 only |  | 0.02 | -0.0112 | [-0.0156, -0.0066] | equivalent |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.01 | -0.0562 | [-0.0589, -0.0535] | not equivalent |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.02 | -0.0562 | [-0.0589, -0.0535] | not equivalent |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.01 | -0.0654 | [-0.0684, -0.0623] | not equivalent |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.02 | -0.0654 | [-0.0684, -0.0623] | not equivalent |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.01 | -0.0542 | [-0.0575, -0.0512] | not equivalent |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 0.02 | -0.0542 | [-0.0575, -0.0512] | not equivalent |

**Which pairs changed verdict against the old (secondary, mean-per-subject) version (task item 3):** matched by (model, condition A, condition B, margin) against the SECONDARY TOST table below -- exactly **1 of 36** verdicts changes: SleepFM Next-Token, EEG1 only vs EEG3 only, margin 0.01 -- PRIMARY says "inconclusive" (90% CI [0.0005, 0.0101], just barely overlapping the 0.01 boundary), SECONDARY said "equivalent" (90% CI [-0.0004, 0.0087], entirely inside 0.01). Every other verdict is unchanged between the two statistics, at both margins.

**SECONDARY statistic (mean per-subject macro F1):**

*No marks: equivalence verdicts, not a ranked comparison.*
<!-- spec: items=TOST verdicts (secondary, mean per-subject); marks=none; note=equivalence verdicts, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | Margin | Point diff | 90% CI (mean macro F1 diff) | Verdict |
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

### D.4: Friedman test across EEG1/EEG2/EEG3 (secondary statistic)
Per-subject macro F1, same as D.2's secondary statistic -- tests whether the three channels' typical (median) subject differs, not whether their means differ.

*No marks: omnibus test, not a ranked comparison.*
<!-- spec: items=Friedman test results; marks=none; note=omnibus test, not a ranked comparison -->
| Model | n subjects | Friedman statistic | p-value |
|---|---|---|---|
| SleepFM From-Scratch | 2056 | 19.682 | 5.32e-05 |
| SleepFM Next-Token | 2056 | 37.561 | 6.98e-09 |
| BIOT | 2056 | 32.152 | 1.04e-07 |


**Holm-adjusted post-hoc pairwise Wilcoxon (per model), with the Hodges-Lehmann estimate:**

*No marks: post-hoc comparisons, not a ranked comparison.*
<!-- spec: items=post-hoc pairwise comparisons; marks=none; note=post-hoc comparisons, not a ranked comparison -->
| Model | Condition A | Condition B | n subjects | Median diff macro F1 | Hodges-Lehmann diff macro F1 | Wilcoxon p | Rank-biserial | Holm p |
|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only | 2056 | -0.0051 | -0.0050 | 0.0001 | -0.099 | 0.0003 |
| SleepFM From-Scratch | EEG1 only | EEG3 only | 2056 | 0.0022 | 0.0012 | 0.4594 | 0.019 | 0.4594 |
| SleepFM From-Scratch | EEG2 only | EEG3 only | 2056 | 0.0058 | 0.0041 | 0.0099 | 0.066 | 0.0198 |
| SleepFM Next-Token | EEG1 only | EEG2 only | 2056 | -0.0029 | -0.0023 | 0.0480 | -0.050 | 0.0480 |
| SleepFM Next-Token | EEG1 only | EEG3 only | 2056 | 0.0052 | 0.0041 | 0.0037 | 0.074 | 0.0074 |
| SleepFM Next-Token | EEG2 only | EEG3 only | 2056 | 0.0076 | 0.0059 | 1.38e-05 | 0.111 | 4.15e-05 |
| BIOT | EEG1 only | EEG2 only | 2056 | 0.0051 | 0.0043 | 0.0010 | 0.084 | 0.0020 |
| BIOT | EEG1 only | EEG3 only | 2056 | -0.0006 | -0.0024 | 0.1587 | -0.036 | 0.1587 |
| BIOT | EEG2 only | EEG3 only | 2056 | -0.0071 | -0.0081 | 3.97e-07 | -0.129 | 1.19e-06 |


### D.5: per-class recall and pooled per-class F1
**Per-subject RECALL (secondary statistic), mean paired difference with a bootstrap CI as the headline (never the median -- the median is routinely exactly 0 for a stage most subjects tie on, which would be a misleading headline; median/Wilcoxon/Holm kept as the secondary rank-based test). N1, N3, REM:**

*No marks: per-class comparisons, not a ranked comparison.*
<!-- spec: items=per-class recall comparisons; marks=none; note=per-class comparisons, not a ranked comparison -->
| Model | Stage | Condition A | Condition B | Reference | n subjects | n dropped (undefined) | Mean diff recall | Bootstrap 95% CI | Median diff recall | Wilcoxon p | Rank-biserial | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | N1 | EEG1 only | EEG2 only |  | 2035 | 21 | -0.0344 | [-0.0395, -0.0290] | -0.0312 | 6.94e-43 | -0.362 | 2.08e-42 |
| SleepFM From-Scratch | N1 | EEG1 only | EEG3 only |  | 2035 | 21 | -0.0309 | [-0.0372, -0.0242] | -0.0253 | 9.61e-19 | -0.231 | 1.92e-18 |
| SleepFM From-Scratch | N1 | EEG2 only | EEG3 only |  | 2035 | 21 | 0.0035 | [-0.0029, 0.0101] | 0.0098 | 0.0164 | 0.063 | 0.0164 |
| SleepFM From-Scratch | N1 | EEG1 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.1139 | [-0.1188, -0.1087] | -0.1111 | 3.01e-232 | -0.847 | 1.81e-231 |
| SleepFM From-Scratch | N1 | EEG2 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0795 | [-0.0840, -0.0749] | -0.0714 | 2.77e-171 | -0.731 | 1.39e-170 |
| SleepFM From-Scratch | N1 | EEG3 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0830 | [-0.0884, -0.0775] | -0.0781 | 9.99e-153 | -0.687 | 3.99e-152 |
| SleepFM From-Scratch | N3 | EEG1 only | EEG2 only |  | 1885 | 171 | -0.0290 | [-0.0419, -0.0161] | 0.0000 | 4.10e-06 | -0.135 | 1.23e-05 |
| SleepFM From-Scratch | N3 | EEG1 only | EEG3 only |  | 1885 | 171 | 0.0199 | [0.0048, 0.0348] | 0.0000 | 0.0206 | 0.067 | 0.0206 |
| SleepFM From-Scratch | N3 | EEG2 only | EEG3 only |  | 1885 | 171 | 0.0489 | [0.0345, 0.0633] | 0.0000 | 1.31e-09 | 0.175 | 5.25e-09 |
| SleepFM From-Scratch | N3 | EEG1 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | -0.0306 | [-0.0426, -0.0187] | 0.0000 | 1.21e-13 | -0.219 | 6.06e-13 |
| SleepFM From-Scratch | N3 | EEG2 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | -0.0016 | [-0.0130, 0.0108] | 0.0000 | 0.0024 | -0.090 | 0.0048 |
| SleepFM From-Scratch | N3 | EEG3 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | -0.0505 | [-0.0637, -0.0375] | -0.0085 | 5.52e-20 | -0.268 | 3.31e-19 |
| SleepFM From-Scratch | REM | EEG1 only | EEG2 only |  | 2012 | 44 | 0.0072 | [-0.0015, 0.0161] | 0.0000 | 0.8624 | 0.005 | 1.0000 |
| SleepFM From-Scratch | REM | EEG1 only | EEG3 only |  | 2012 | 44 | 0.0033 | [-0.0071, 0.0132] | 0.0000 | 0.2816 | 0.031 | 0.8448 |
| SleepFM From-Scratch | REM | EEG2 only | EEG3 only |  | 2012 | 44 | -0.0040 | [-0.0141, 0.0060] | 0.0000 | 0.6125 | 0.014 | 1.0000 |
| SleepFM From-Scratch | REM | EEG1 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | -0.0348 | [-0.0425, -0.0275] | 0.0000 | 2.72e-16 | -0.239 | 1.09e-15 |
| SleepFM From-Scratch | REM | EEG2 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | -0.0421 | [-0.0498, -0.0347] | 0.0000 | 7.00e-22 | -0.280 | 3.50e-21 |
| SleepFM From-Scratch | REM | EEG3 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | -0.0381 | [-0.0455, -0.0309] | 0.0000 | 2.91e-24 | -0.295 | 1.75e-23 |
| SleepFM Next-Token | N1 | EEG1 only | EEG2 only |  | 2035 | 21 | -0.0054 | [-0.0106, -0.0001] | -0.0122 | 2.89e-05 | -0.111 | 2.89e-05 |
| SleepFM Next-Token | N1 | EEG1 only | EEG3 only |  | 2035 | 21 | 0.0369 | [0.0305, 0.0424] | 0.0364 | 2.12e-35 | 0.326 | 4.25e-35 |
| SleepFM Next-Token | N1 | EEG2 only | EEG3 only |  | 2035 | 21 | 0.0423 | [0.0361, 0.0481] | 0.0476 | 2.82e-57 | 0.417 | 8.47e-57 |
| SleepFM Next-Token | N1 | EEG1 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0440 | [-0.0487, -0.0396] | -0.0441 | 2.64e-77 | -0.490 | 1.32e-76 |
| SleepFM Next-Token | N1 | EEG2 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0386 | [-0.0429, -0.0342] | -0.0303 | 1.34e-58 | -0.427 | 5.37e-58 |
| SleepFM Next-Token | N1 | EEG3 only | 3-channel, patience 10 | patience 10 | 2035 | 21 | -0.0809 | [-0.0855, -0.0756] | -0.0741 | 8.83e-171 | -0.729 | 5.30e-170 |
| SleepFM Next-Token | N3 | EEG1 only | EEG2 only |  | 1885 | 171 | -0.0253 | [-0.0378, -0.0133] | 0.0000 | 2.57e-07 | -0.153 | 1.03e-06 |
| SleepFM Next-Token | N3 | EEG1 only | EEG3 only |  | 1885 | 171 | -0.0183 | [-0.0317, -0.0044] | 0.0000 | 8.10e-05 | -0.117 | 0.0002 |
| SleepFM Next-Token | N3 | EEG2 only | EEG3 only |  | 1885 | 171 | 0.0070 | [-0.0057, 0.0201] | 0.0000 | 0.7503 | -0.009 | 1.0000 |
| SleepFM Next-Token | N3 | EEG1 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | 0.0051 | [-0.0057, 0.0158] | 0.0000 | 0.8223 | 0.007 | 1.0000 |
| SleepFM Next-Token | N3 | EEG2 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | 0.0304 | [0.0201, 0.0413] | 0.0000 | 4.70e-10 | 0.188 | 2.35e-09 |
| SleepFM Next-Token | N3 | EEG3 only | 3-channel, patience 10 | patience 10 | 1885 | 171 | 0.0234 | [0.0136, 0.0349] | 0.0000 | 3.22e-11 | 0.199 | 1.93e-10 |
| SleepFM Next-Token | REM | EEG1 only | EEG2 only |  | 2012 | 44 | 0.0041 | [-0.0044, 0.0124] | 0.0000 | 0.8006 | -0.007 | 1.0000 |
| SleepFM Next-Token | REM | EEG1 only | EEG3 only |  | 2012 | 44 | 0.0137 | [0.0046, 0.0224] | 0.0000 | 0.0005 | 0.100 | 0.0022 |
| SleepFM Next-Token | REM | EEG2 only | EEG3 only |  | 2012 | 44 | 0.0096 | [0.0004, 0.0188] | 0.0000 | 1.64e-05 | 0.125 | 8.20e-05 |
| SleepFM Next-Token | REM | EEG1 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | -0.0079 | [-0.0140, -0.0014] | 0.0000 | 0.3476 | -0.028 | 1.0000 |
| SleepFM Next-Token | REM | EEG2 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | -0.0120 | [-0.0190, -0.0050] | 0.0000 | 0.3690 | -0.027 | 1.0000 |
| SleepFM Next-Token | REM | EEG3 only | 3-channel, patience 10 | patience 10 | 2012 | 44 | -0.0216 | [-0.0295, -0.0139] | 0.0000 | 1.01e-07 | -0.159 | 6.08e-07 |
| BIOT | N1 | EEG1 only | EEG2 only |  | 2035 | 21 | 0.0509 | [0.0441, 0.0580] | 0.0471 | 1.11e-52 | 0.399 | 3.34e-52 |
| BIOT | N1 | EEG1 only | EEG3 only |  | 2035 | 21 | 0.0016 | [-0.0073, 0.0100] | 0.0000 | 0.6598 | 0.011 | 0.6598 |
| BIOT | N1 | EEG2 only | EEG3 only |  | 2035 | 21 | -0.0493 | [-0.0580, -0.0408] | -0.0470 | 6.84e-33 | -0.312 | 1.37e-32 |
| BIOT | N1 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2035 | 21 | -0.0879 | [-0.0942, -0.0817] | -0.0753 | 1.63e-136 | -0.651 | 6.50e-136 |
| BIOT | N1 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2035 | 21 | -0.1388 | [-0.1460, -0.1322] | -0.1250 | 2.49e-228 | -0.835 | 1.50e-227 |
| BIOT | N1 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2035 | 21 | -0.0895 | [-0.0966, -0.0828] | -0.0698 | 2.11e-143 | -0.670 | 1.05e-142 |
| BIOT | N3 | EEG1 only | EEG2 only |  | 1885 | 171 | -0.0359 | [-0.0480, -0.0232] | 0.0000 | 5.08e-13 | -0.208 | 2.03e-12 |
| BIOT | N3 | EEG1 only | EEG3 only |  | 1885 | 171 | -0.0714 | [-0.0866, -0.0565] | -0.0256 | 9.77e-27 | -0.307 | 5.86e-26 |
| BIOT | N3 | EEG2 only | EEG3 only |  | 1885 | 171 | -0.0356 | [-0.0504, -0.0209] | 0.0000 | 5.19e-10 | -0.181 | 1.56e-09 |
| BIOT | N3 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 1885 | 171 | -0.0356 | [-0.0471, -0.0244] | 0.0000 | 1.87e-07 | -0.152 | 3.74e-07 |
| BIOT | N3 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 1885 | 171 | 0.0002 | [-0.0111, 0.0120] | 0.0000 | 0.1650 | 0.041 | 0.1650 |
| BIOT | N3 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 1885 | 171 | 0.0358 | [0.0251, 0.0461] | 0.0088 | 1.56e-22 | 0.288 | 7.81e-22 |
| BIOT | REM | EEG1 only | EEG2 only |  | 2012 | 44 | 0.0254 | [0.0168, 0.0345] | 0.0062 | 7.49e-05 | 0.104 | 0.0002 |
| BIOT | REM | EEG1 only | EEG3 only |  | 2012 | 44 | 0.0150 | [0.0038, 0.0261] | 0.0000 | 0.0145 | 0.064 | 0.0289 |
| BIOT | REM | EEG2 only | EEG3 only |  | 2012 | 44 | -0.0104 | [-0.0225, 0.0011] | 0.0000 | 0.1532 | -0.037 | 0.1532 |
| BIOT | REM | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2012 | 44 | -0.0580 | [-0.0659, -0.0506] | -0.0398 | 5.17e-65 | -0.448 | 2.07e-64 |
| BIOT | REM | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2012 | 44 | -0.0834 | [-0.0922, -0.0748] | -0.0444 | 4.27e-82 | -0.505 | 2.56e-81 |
| BIOT | REM | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2012 | 44 | -0.0730 | [-0.0814, -0.0650] | -0.0392 | 2.45e-75 | -0.484 | 1.22e-74 |

Holm adjustment is applied within each (model, stage) pair across its 6 comparisons. A subject is dropped from a pair when that stage is undefined (absent from that subject's night) in either condition being compared.

**Wake, N2:**

*No marks: per-class comparisons, not a ranked comparison.*
<!-- spec: items=per-class recall comparisons; marks=none; note=per-class comparisons, not a ranked comparison -->
| Model | Stage | Condition A | Condition B | Reference | n subjects | n dropped (undefined) | Mean diff recall | Bootstrap 95% CI | Median diff recall | Wilcoxon p | Rank-biserial | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | Wake | EEG1 only | EEG2 only |  | 2056 | 0 | -0.0118 | [-0.0152, -0.0084] | -0.0048 | 1.82e-33 | -0.316 | 5.47e-33 |
| SleepFM From-Scratch | Wake | EEG1 only | EEG3 only |  | 2056 | 0 | -0.0131 | [-0.0171, -0.0094] | -0.0042 | 7.31e-24 | -0.262 | 1.46e-23 |
| SleepFM From-Scratch | Wake | EEG2 only | EEG3 only |  | 2056 | 0 | -0.0014 | [-0.0049, 0.0023] | 0.0000 | 0.5159 | 0.017 | 0.5159 |
| SleepFM From-Scratch | Wake | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0293 | [-0.0328, -0.0262] | -0.0107 | 6.74e-145 | -0.672 | 4.05e-144 |
| SleepFM From-Scratch | Wake | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0176 | [-0.0205, -0.0147] | -0.0045 | 1.11e-58 | -0.428 | 4.45e-58 |
| SleepFM From-Scratch | Wake | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0162 | [-0.0190, -0.0133] | -0.0051 | 1.93e-60 | -0.431 | 9.67e-60 |
| SleepFM From-Scratch | N2 | EEG1 only | EEG2 only |  | 2056 | 0 | 0.0330 | [0.0264, 0.0392] | 0.0284 | 1.43e-40 | 0.343 | 8.58e-40 |
| SleepFM From-Scratch | N2 | EEG1 only | EEG3 only |  | 2056 | 0 | 0.0235 | [0.0161, 0.0311] | 0.0299 | 6.98e-23 | 0.252 | 2.79e-22 |
| SleepFM From-Scratch | N2 | EEG2 only | EEG3 only |  | 2056 | 0 | -0.0095 | [-0.0167, -0.0024] | -0.0040 | 0.1016 | -0.042 | 0.1016 |
| SleepFM From-Scratch | N2 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0068 | [-0.0129, -0.0011] | 0.0093 | 0.0065 | 0.070 | 0.0129 |
| SleepFM From-Scratch | N2 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0398 | [-0.0458, -0.0343] | -0.0173 | 3.60e-28 | -0.282 | 1.80e-27 |
| SleepFM From-Scratch | N2 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0303 | [-0.0358, -0.0250] | -0.0178 | 7.55e-22 | -0.246 | 2.27e-21 |
| SleepFM Next-Token | Wake | EEG1 only | EEG2 only |  | 2056 | 0 | -0.0048 | [-0.0086, -0.0008] | -0.0031 | 1.53e-11 | -0.177 | 3.06e-11 |
| SleepFM Next-Token | Wake | EEG1 only | EEG3 only |  | 2056 | 0 | -0.0132 | [-0.0176, -0.0087] | -0.0064 | 8.69e-31 | -0.301 | 3.47e-30 |
| SleepFM Next-Token | Wake | EEG2 only | EEG3 only |  | 2056 | 0 | -0.0084 | [-0.0124, -0.0043] | -0.0031 | 8.90e-12 | -0.179 | 2.67e-11 |
| SleepFM Next-Token | Wake | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0210 | [-0.0248, -0.0175] | -0.0072 | 2.49e-73 | -0.475 | 1.49e-72 |
| SleepFM Next-Token | Wake | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0162 | [-0.0198, -0.0129] | -0.0031 | 1.51e-34 | -0.325 | 7.56e-34 |
| SleepFM Next-Token | Wake | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0078 | [-0.0111, -0.0047] | 0.0000 | 0.0047 | -0.075 | 0.0047 |
| SleepFM Next-Token | N2 | EEG1 only | EEG2 only |  | 2056 | 0 | 0.0214 | [0.0148, 0.0284] | 0.0142 | 1.12e-12 | 0.182 | 3.35e-12 |
| SleepFM Next-Token | N2 | EEG1 only | EEG3 only |  | 2056 | 0 | 0.0011 | [-0.0068, 0.0093] | 0.0035 | 0.4005 | 0.022 | 0.4005 |
| SleepFM Next-Token | N2 | EEG2 only | EEG3 only |  | 2056 | 0 | -0.0204 | [-0.0286, -0.0122] | -0.0107 | 9.67e-08 | -0.136 | 1.93e-07 |
| SleepFM Next-Token | N2 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0421 | [-0.0481, -0.0362] | -0.0191 | 7.11e-40 | -0.339 | 2.84e-39 |
| SleepFM Next-Token | N2 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0635 | [-0.0699, -0.0575] | -0.0349 | 8.13e-100 | -0.544 | 4.88e-99 |
| SleepFM Next-Token | N2 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | 0 | -0.0432 | [-0.0490, -0.0373] | -0.0218 | 6.84e-46 | -0.365 | 3.42e-45 |
| BIOT | Wake | EEG1 only | EEG2 only |  | 2056 | 0 | -0.0114 | [-0.0154, -0.0073] | -0.0082 | 2.48e-19 | -0.232 | 7.44e-19 |
| BIOT | Wake | EEG1 only | EEG3 only |  | 2056 | 0 | -0.0111 | [-0.0150, -0.0072] | -0.0062 | 1.49e-12 | -0.182 | 2.99e-12 |
| BIOT | Wake | EEG2 only | EEG3 only |  | 2056 | 0 | 0.0003 | [-0.0035, 0.0041] | 0.0015 | 0.0625 | 0.048 | 0.0625 |
| BIOT | Wake | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0300 | [-0.0330, -0.0270] | -0.0140 | 9.55e-99 | -0.544 | 5.73e-98 |
| BIOT | Wake | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0186 | [-0.0218, -0.0155] | -0.0052 | 4.52e-33 | -0.311 | 1.81e-32 |
| BIOT | Wake | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0189 | [-0.0217, -0.0161] | -0.0078 | 4.32e-52 | -0.392 | 2.16e-51 |
| BIOT | N2 | EEG1 only | EEG2 only |  | 2056 | 0 | 0.0067 | [0.0008, 0.0127] | 0.0036 | 0.0157 | 0.062 | 0.0157 |
| BIOT | N2 | EEG1 only | EEG3 only |  | 2056 | 0 | 0.0097 | [0.0024, 0.0171] | 0.0135 | 5.80e-07 | 0.128 | 1.74e-06 |
| BIOT | N2 | EEG2 only | EEG3 only |  | 2056 | 0 | 0.0031 | [-0.0038, 0.0106] | 0.0105 | 0.0068 | 0.069 | 0.0135 |
| BIOT | N2 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0629 | [-0.0679, -0.0578] | -0.0439 | 2.10e-128 | -0.617 | 1.05e-127 |
| BIOT | N2 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0695 | [-0.0753, -0.0638] | -0.0468 | 7.18e-128 | -0.616 | 2.87e-127 |
| BIOT | N2 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | 0 | -0.0726 | [-0.0784, -0.0668] | -0.0591 | 1.43e-142 | -0.651 | 8.59e-142 |


**Pooled per-class F1 (epoch-pooled, same pooling logic as the PRIMARY macro-F1 statistic, just not averaged across classes). No subject is ever dropped here for "undefined" -- pooling is at the epoch level, so a subject with zero occurrences of a class still contributes correctly (0 TP, possibly nonzero FP) to the pooled counts; `n` is always the full common-subject count:**

*No marks: per-class comparisons, not a ranked comparison.*
<!-- spec: items=pooled per-class F1 comparisons; marks=none; note=per-class comparisons, not a ranked comparison -->
| Model | Stage | Condition A | Condition B | Reference | n subjects | Pooled F1 diff (A-B) | Bootstrap 95% CI |
|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | Wake | EEG1 only | EEG2 only |  | 2056 | -0.0050 | [-0.0081, -0.0018] |
| SleepFM From-Scratch | Wake | EEG1 only | EEG3 only |  | 2056 | 0.0007 | [-0.0031, 0.0046] |
| SleepFM From-Scratch | Wake | EEG2 only | EEG3 only |  | 2056 | 0.0057 | [0.0018, 0.0098] |
| SleepFM From-Scratch | Wake | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0255 | [-0.0284, -0.0226] |
| SleepFM From-Scratch | Wake | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0205 | [-0.0234, -0.0178] |
| SleepFM From-Scratch | Wake | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0262 | [-0.0291, -0.0233] |
| SleepFM From-Scratch | N1 | EEG1 only | EEG2 only |  | 2056 | -0.0325 | [-0.0382, -0.0262] |
| SleepFM From-Scratch | N1 | EEG1 only | EEG3 only |  | 2056 | -0.0199 | [-0.0276, -0.0119] |
| SleepFM From-Scratch | N1 | EEG2 only | EEG3 only |  | 2056 | 0.0125 | [0.0053, 0.0200] |
| SleepFM From-Scratch | N1 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.1064 | [-0.1124, -0.1003] |
| SleepFM From-Scratch | N1 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0739 | [-0.0787, -0.0690] |
| SleepFM From-Scratch | N1 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0864 | [-0.0922, -0.0806] |
| SleepFM From-Scratch | N2 | EEG1 only | EEG2 only |  | 2056 | 0.0131 | [0.0087, 0.0174] |
| SleepFM From-Scratch | N2 | EEG1 only | EEG3 only |  | 2056 | 0.0080 | [0.0028, 0.0133] |
| SleepFM From-Scratch | N2 | EEG2 only | EEG3 only |  | 2056 | -0.0051 | [-0.0100, 0.0001] |
| SleepFM From-Scratch | N2 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0213 | [-0.0253, -0.0176] |
| SleepFM From-Scratch | N2 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0343 | [-0.0382, -0.0306] |
| SleepFM From-Scratch | N2 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0293 | [-0.0331, -0.0255] |
| SleepFM From-Scratch | N3 | EEG1 only | EEG2 only |  | 2056 | 0.0065 | [-0.0025, 0.0152] |
| SleepFM From-Scratch | N3 | EEG1 only | EEG3 only |  | 2056 | -0.0166 | [-0.0263, -0.0065] |
| SleepFM From-Scratch | N3 | EEG2 only | EEG3 only |  | 2056 | -0.0232 | [-0.0328, -0.0128] |
| SleepFM From-Scratch | N3 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0292 | [-0.0372, -0.0218] |
| SleepFM From-Scratch | N3 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0358 | [-0.0437, -0.0279] |
| SleepFM From-Scratch | N3 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0126 | [-0.0204, -0.0053] |
| SleepFM From-Scratch | REM | EEG1 only | EEG2 only |  | 2056 | 0.0144 | [0.0063, 0.0227] |
| SleepFM From-Scratch | REM | EEG1 only | EEG3 only |  | 2056 | 0.0020 | [-0.0083, 0.0122] |
| SleepFM From-Scratch | REM | EEG2 only | EEG3 only |  | 2056 | -0.0125 | [-0.0220, -0.0021] |
| SleepFM From-Scratch | REM | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0533 | [-0.0604, -0.0459] |
| SleepFM From-Scratch | REM | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0677 | [-0.0748, -0.0610] |
| SleepFM From-Scratch | REM | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0552 | [-0.0628, -0.0480] |
| SleepFM Next-Token | Wake | EEG1 only | EEG2 only |  | 2056 | -0.0057 | [-0.0089, -0.0022] |
| SleepFM Next-Token | Wake | EEG1 only | EEG3 only |  | 2056 | 0.0005 | [-0.0037, 0.0050] |
| SleepFM Next-Token | Wake | EEG2 only | EEG3 only |  | 2056 | 0.0062 | [0.0019, 0.0106] |
| SleepFM Next-Token | Wake | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0207 | [-0.0237, -0.0177] |
| SleepFM Next-Token | Wake | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0151 | [-0.0181, -0.0122] |
| SleepFM Next-Token | Wake | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0212 | [-0.0247, -0.0178] |
| SleepFM Next-Token | N1 | EEG1 only | EEG2 only |  | 2056 | -0.0077 | [-0.0132, -0.0024] |
| SleepFM Next-Token | N1 | EEG1 only | EEG3 only |  | 2056 | 0.0423 | [0.0355, 0.0492] |
| SleepFM Next-Token | N1 | EEG2 only | EEG3 only |  | 2056 | 0.0500 | [0.0438, 0.0562] |
| SleepFM Next-Token | N1 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0386 | [-0.0434, -0.0337] |
| SleepFM Next-Token | N1 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0310 | [-0.0352, -0.0264] |
| SleepFM Next-Token | N1 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0810 | [-0.0863, -0.0759] |
| SleepFM Next-Token | N2 | EEG1 only | EEG2 only |  | 2056 | 0.0115 | [0.0073, 0.0159] |
| SleepFM Next-Token | N2 | EEG1 only | EEG3 only |  | 2056 | 0.0046 | [-0.0007, 0.0099] |
| SleepFM Next-Token | N2 | EEG2 only | EEG3 only |  | 2056 | -0.0069 | [-0.0122, -0.0016] |
| SleepFM Next-Token | N2 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0246 | [-0.0284, -0.0210] |
| SleepFM Next-Token | N2 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0361 | [-0.0401, -0.0325] |
| SleepFM Next-Token | N2 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0292 | [-0.0331, -0.0256] |
| SleepFM Next-Token | N3 | EEG1 only | EEG2 only |  | 2056 | 0.0027 | [-0.0057, 0.0104] |
| SleepFM Next-Token | N3 | EEG1 only | EEG3 only |  | 2056 | -0.0147 | [-0.0244, -0.0046] |
| SleepFM Next-Token | N3 | EEG2 only | EEG3 only |  | 2056 | -0.0174 | [-0.0271, -0.0075] |
| SleepFM Next-Token | N3 | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0355 | [-0.0421, -0.0291] |
| SleepFM Next-Token | N3 | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0383 | [-0.0450, -0.0315] |
| SleepFM Next-Token | N3 | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0208 | [-0.0292, -0.0135] |
| SleepFM Next-Token | REM | EEG1 only | EEG2 only |  | 2056 | 0.0245 | [0.0142, 0.0353] |
| SleepFM Next-Token | REM | EEG1 only | EEG3 only |  | 2056 | -0.0066 | [-0.0184, 0.0055] |
| SleepFM Next-Token | REM | EEG2 only | EEG3 only |  | 2056 | -0.0310 | [-0.0428, -0.0192] |
| SleepFM Next-Token | REM | EEG1 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0460 | [-0.0553, -0.0371] |
| SleepFM Next-Token | REM | EEG2 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0705 | [-0.0804, -0.0616] |
| SleepFM Next-Token | REM | EEG3 only | 3-channel, patience 10 | patience 10 | 2056 | -0.0395 | [-0.0483, -0.0313] |
| BIOT | Wake | EEG1 only | EEG2 only |  | 2056 | -0.0079 | [-0.0117, -0.0041] |
| BIOT | Wake | EEG1 only | EEG3 only |  | 2056 | -0.0112 | [-0.0157, -0.0067] |
| BIOT | Wake | EEG2 only | EEG3 only |  | 2056 | -0.0032 | [-0.0077, 0.0014] |
| BIOT | Wake | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0385 | [-0.0420, -0.0350] |
| BIOT | Wake | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0305 | [-0.0341, -0.0271] |
| BIOT | Wake | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0273 | [-0.0303, -0.0243] |
| BIOT | N1 | EEG1 only | EEG2 only |  | 2056 | 0.0004 | [-0.0047, 0.0054] |
| BIOT | N1 | EEG1 only | EEG3 only |  | 2056 | 0.0023 | [-0.0034, 0.0080] |
| BIOT | N1 | EEG2 only | EEG3 only |  | 2056 | 0.0019 | [-0.0040, 0.0078] |
| BIOT | N1 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0534 | [-0.0575, -0.0495] |
| BIOT | N1 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0538 | [-0.0586, -0.0492] |
| BIOT | N1 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0558 | [-0.0599, -0.0517] |
| BIOT | N2 | EEG1 only | EEG2 only |  | 2056 | 0.0130 | [0.0085, 0.0176] |
| BIOT | N2 | EEG1 only | EEG3 only |  | 2056 | 0.0106 | [0.0052, 0.0162] |
| BIOT | N2 | EEG2 only | EEG3 only |  | 2056 | -0.0024 | [-0.0080, 0.0033] |
| BIOT | N2 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0455 | [-0.0491, -0.0419] |
| BIOT | N2 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0585 | [-0.0627, -0.0544] |
| BIOT | N2 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0561 | [-0.0602, -0.0519] |
| BIOT | N3 | EEG1 only | EEG2 only |  | 2056 | 0.0075 | [-0.0001, 0.0148] |
| BIOT | N3 | EEG1 only | EEG3 only |  | 2056 | -0.0146 | [-0.0233, -0.0052] |
| BIOT | N3 | EEG2 only | EEG3 only |  | 2056 | -0.0220 | [-0.0302, -0.0128] |
| BIOT | N3 | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0545 | [-0.0618, -0.0471] |
| BIOT | N3 | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0620 | [-0.0685, -0.0548] |
| BIOT | N3 | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0400 | [-0.0470, -0.0334] |
| BIOT | REM | EEG1 only | EEG2 only |  | 2056 | 0.0329 | [0.0256, 0.0408] |
| BIOT | REM | EEG1 only | EEG3 only |  | 2056 | 0.0024 | [-0.0076, 0.0119] |
| BIOT | REM | EEG2 only | EEG3 only |  | 2056 | -0.0305 | [-0.0404, -0.0204] |
| BIOT | REM | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0894 | [-0.0958, -0.0834] |
| BIOT | REM | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.1223 | [-0.1293, -0.1154] |
| BIOT | REM | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 2056 | -0.0918 | [-0.0990, -0.0848] |


### D.6: models against each other, same subjects, per condition
Secondary statistic (mean per-subject macro F1), unchanged methodology from before this revision.

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


**Pretraining and patience note (exact statement, task item 6/7):** "SleepFM's two encoders' pretraining configs -- `fromscratch_eegonly_fullcohort/.../config.json` and `nexttoken_eegonly_fullcohort/.../config.json` -- both set `split_path` to the SAME file, `sleepfm/configs/dataset_split_fromscratch_fullcohort.json`, whose `pretrain` key lists 2056 subjects. The fine-tuning/test split file, `sleepfm/configs/dataset_split_fold5_v1.json`, has 5 folds (`fold_0`-`fold_4`) whose `test` subjects union to exactly 2056 (411+411+411+411+412). The exact-ID set intersection of these two subject lists is the full 2056 -- every fine-tuning/test subject is also a pretraining subject (2056/2056, set equality, checked by direct ID comparison, not by count alone). Pretraining itself is self-supervised (contrastive for From-Scratch, next-token-prediction for Next-Token) and uses no sleep-stage label, so this is subject-level overlap, not label leakage -- but it is real overlap, and is stated as such rather than elided." Verification: loaded both JSON split files directly, took `os.path.basename(path).split('.')[0]` as the subject ID from each HDF5 path entry, and compared the two ID sets with Python's set operations (`&`, `==`) -- reproducible via the one-off check recorded in this task's own log, not a script committed to the repo (no new script was needed; this uses only the two existing config/split JSON files already in `sleepfm/configs/`). BIOT's encoder is pretrained externally, before this project, on EEG-SHHS+PREST (5M MGH resting-EEG + 5M Sleep Heart Health Study samples -- a sleep-PSG cohort comparable to MESA, but not MESA itself, so no subject overlap is possible for BIOT). Patience: SleepFM uses patience 10 for every single-channel run and the matched-patience 3-channel reference, patience 50 for its 3-channel benchmark run. BIOT uses patience 10 for its single-channel runs (confirmed via `biot_fullcohort_patience10.slurm` in each run's sacct `SubmitLine`) and patience 20 for its only 3-channel run (it has no patience-10 3-channel run, so the patience-20 run serves as its reference in D.2/D.3 above, labeled as such throughout).

### D.7: fold-level sign counts
Both forms, labeled: "subject-averaged" (mean of each fold's subjects' own macro F1 -- the secondary statistic, fold-restricted) and "fold macro F1" (the epoch-pooled per-fold statistic, same methodology as results/02/04/06 and this file's own D.1 primary per-fold values). These can disagree in sign on individual folds (see the nonlinearity note under D.1).

*No marks: fold-level differences, not a ranked comparison.*
<!-- spec: items=fold-level sign counts; marks=none; note=fold-level differences, not a ranked comparison -->
| Model | Condition A | Condition B | Reference | n folds | Per-fold diff, subject-averaged (A - B) | Sign counts (subject-averaged) | Per-fold diff, fold macro F1 (A - B) | Sign counts (fold macro F1) |
|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG1 only | EEG2 only |  | 5 | 0.0099, -0.0106, -0.0176, 0.0058, 0.0064 | +3/-2/0:0 | 0.0101, -0.0106, -0.0147, 0.0062, 0.0051 | +3/-2/0:0 |
| SleepFM From-Scratch | EEG1 only | EEG3 only |  | 5 | -0.0018, -0.0021, -0.0264, 0.0058, 0.0183 | +2/-3/0:0 | -0.0067, -0.0028, -0.0337, 0.0011, 0.0165 | +2/-3/0:0 |
| SleepFM From-Scratch | EEG2 only | EEG3 only |  | 5 | -0.0117, 0.0085, -0.0088, -0.0000, 0.0119 | +2/-3/0:0 | -0.0167, 0.0078, -0.0190, -0.0051, 0.0115 | +2/-3/0:0 |
| SleepFM From-Scratch | EEG1 only | 3-channel, patience 10 | patience 10 | 5 | -0.0232, -0.0520, -0.0649, -0.0398, -0.0435 | +0/-5/0:0 | -0.0240, -0.0527, -0.0714, -0.0423, -0.0454 | +0/-5/0:0 |
| SleepFM From-Scratch | EEG2 only | 3-channel, patience 10 | patience 10 | 5 | -0.0331, -0.0414, -0.0472, -0.0457, -0.0500 | +0/-5/0:0 | -0.0340, -0.0422, -0.0567, -0.0485, -0.0504 | +0/-5/0:0 |
| SleepFM From-Scratch | EEG3 only | 3-channel, patience 10 | patience 10 | 5 | -0.0214, -0.0499, -0.0384, -0.0456, -0.0619 | +0/-5/0:0 | -0.0173, -0.0499, -0.0377, -0.0434, -0.0619 | +0/-5/0:0 |
| SleepFM Next-Token | EEG1 only | EEG2 only |  | 5 | -0.0002, 0.0044, 0.0044, -0.0070, 0.0130 | +3/-2/0:0 | -0.0003, 0.0100, 0.0043, -0.0090, 0.0192 | +3/-2/0:0 |
| SleepFM Next-Token | EEG1 only | EEG3 only |  | 5 | -0.0020, 0.0042, 0.0085, -0.0048, 0.0144 | +3/-2/0:0 | -0.0020, 0.0085, 0.0073, -0.0059, 0.0175 | +3/-2/0:0 |
| SleepFM Next-Token | EEG2 only | EEG3 only |  | 5 | -0.0017, -0.0002, 0.0041, 0.0022, 0.0014 | +3/-2/0:0 | -0.0017, -0.0016, 0.0030, 0.0031, -0.0017 | +2/-3/0:0 |
| SleepFM Next-Token | EEG1 only | 3-channel, patience 10 | patience 10 | 5 | -0.0260, -0.0253, -0.0341, -0.0363, -0.0302 | +0/-5/0:0 | -0.0310, -0.0245, -0.0395, -0.0406, -0.0301 | +0/-5/0:0 |
| SleepFM Next-Token | EEG2 only | 3-channel, patience 10 | patience 10 | 5 | -0.0257, -0.0297, -0.0386, -0.0293, -0.0432 | +0/-5/0:0 | -0.0307, -0.0345, -0.0438, -0.0316, -0.0493 | +0/-5/0:0 |
| SleepFM Next-Token | EEG3 only | 3-channel, patience 10 | patience 10 | 5 | -0.0240, -0.0296, -0.0426, -0.0315, -0.0446 | +0/-5/0:0 | -0.0290, -0.0330, -0.0468, -0.0347, -0.0476 | +0/-5/0:0 |
| BIOT | EEG1 only | EEG2 only |  | 5 | 0.0110, 0.0074, 0.0139, 0.0031, 0.0059 | +5/-0/0:0 | 0.0081, 0.0032, 0.0173, 0.0052, 0.0110 | +5/-0/0:0 |
| BIOT | EEG1 only | EEG3 only |  | 5 | -0.0022, -0.0044, -0.0065, -0.0074, 0.0001 | +1/-4/0:0 | -0.0015, -0.0089, -0.0028, -0.0044, 0.0067 | +1/-4/0:0 |
| BIOT | EEG2 only | EEG3 only |  | 5 | -0.0131, -0.0117, -0.0204, -0.0104, -0.0058 | +0/-5/0:0 | -0.0097, -0.0121, -0.0201, -0.0096, -0.0043 | +0/-5/0:0 |
| BIOT | EEG1 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0617, -0.0550, -0.0597, -0.0555, -0.0611 | +0/-5/0:0 | -0.0629, -0.0551, -0.0537, -0.0557, -0.0552 | +0/-5/0:0 |
| BIOT | EEG2 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0727, -0.0624, -0.0736, -0.0586, -0.0671 | +0/-5/0:0 | -0.0710, -0.0583, -0.0710, -0.0608, -0.0662 | +0/-5/0:0 |
| BIOT | EEG3 only | 3-channel benchmark | patience 20 (BIOT has no patience-10 3-channel run) | 5 | -0.0596, -0.0507, -0.0532, -0.0481, -0.0612 | +0/-5/0:0 | -0.0613, -0.0462, -0.0509, -0.0512, -0.0619 | +0/-5/0:0 |

A two-sided Wilcoxon signed-rank test on 5 paired fold values cannot reach below p = 0.0625 (2/2^5), even when all 5 folds agree in sign -- no p-value is reported at this level for that reason; only the fold-level differences and sign counts above.

## Part E: compute profile

### E1: parameters and checkpoint size
No marks (fixed from the first version of this report, which bolded a column that doesn't change between the only two rows it would ever compare -- a reference/architectural-property table, not a head-to-head ranking).

*No marks: architectural properties, not a ranked comparison.*
<!-- spec: items=checkpoint parameters and size; marks=none; note=architectural properties, not a ranked comparison -->
| Model | Condition | Total params | Trainable params | Encoder params | Head params | Checkpoint size (MiB) | Weights-only size (MiB) | fp16 estimate (MiB) | Note |
|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | 1 channel | 7,019,179 | 2,170,629 | 4,848,550 | 2,170,629 | 45.301 | 26.776 | 13.388 | encoder is frozen during fine-tuning; only the head is trainable. From-Scratch encoder checkpoint also stores optimizer (Adam) state and LR scheduler state alongside the weights |
| SleepFM From-Scratch | 3 channels | 7,019,179 | 2,170,629 | 4,848,550 | 2,170,629 | 45.301 | 26.776 | 13.388 | encoder is frozen during fine-tuning; only the head is trainable. From-Scratch encoder checkpoint also stores optimizer (Adam) state and LR scheduler state alongside the weights |
| SleepFM Next-Token | 1 channel | 7,019,179 | 2,170,629 | 4,848,550 | 2,170,629 | 64.551 | 26.776 | 13.388 | encoder is frozen during fine-tuning; only the head is trainable. Next-Token encoder checkpoint also stores optimizer (Adam) state, LR scheduler state, AND a small next-token-prediction pretext head (head_state_dict, ~0.25MiB, not used at fine-tuning time) alongside the weights |
| SleepFM Next-Token | 3 channels | 7,019,179 | 2,170,629 | 4,848,550 | 2,170,629 | 64.551 | 26.776 | 13.388 | encoder is frozen during fine-tuning; only the head is trainable. Next-Token encoder checkpoint also stores optimizer (Adam) state, LR scheduler state, AND a small next-token-prediction pretext head (head_state_dict, ~0.25MiB, not used at fine-tuning time) alongside the weights |
| BIOT | 1 channel | 3,443,991 | 3,443,991 | n/a | n/a | 13.159 | 13.138 | 6.569 | single end-to-end checkpoint, fully fine-tuned, bare state_dict (no optimizer/scheduler state saved) |
| BIOT | 3 channels | 3,443,991 | 3,443,991 | n/a | n/a | 13.159 | 13.138 | 6.569 | single end-to-end checkpoint, fully fine-tuned, bare state_dict (no optimizer/scheduler state saved) |

SleepFM's SetTransformer encoder and BIOT's encoder both have channel-count-invariant parameter counts (shared per-channel weights for SleepFM; BIOT's published architecture always uses a fixed 18-channel positional embedding regardless of how many channels are actually fed in) -- this is why `total_params` does not change between 1 and 3 channels for any model here; it is a real architectural property, not an error.

**What else is in the checkpoint files (checked directly, task item 7):** BIOT's checkpoint and SleepFM's fine-tuned head checkpoint are both bare `state_dict`s -- weights-only size equals file size almost exactly (BIOT: 13.138MiB weights vs. 13.159MiB file). SleepFM's PRETRAINING encoder checkpoints are much bigger than their weights: From-Scratch's `best.pt` also stores `epoch`, `temperature`, an Adam `optim_dict`, and an LR `scheduler_dict` (18.50MiB weights inside a 37.01MiB file); Next-Token's additionally stores a small next-token-prediction pretext `head_state_dict` (~0.25MiB, not used at fine-tuning time) on top of the same optimizer/scheduler state (18.50MiB weights inside a 56.26MiB file). The weights-only column above sums just the encoder's and head's actual `state_dict` tensors; the gap to the checkpoint-size column is this extra saved state.

### E2: FLOPs per epoch and per night
**Corrected real input shape (task item 5):** verified directly against the pretraining checkpoints' own `config.json` (`sampling_duration=5` [minutes], `sampling_freq=128` [Hz] for both From-Scratch and Next-Token) and `sleepfm/models/dataset.py`'s `SetTransformerDataset.samples_per_chunk = sampling_duration*60*sampling_freq` -- the SleepFM encoder's real forward call processes a **5-minute chunk (38400 samples)**, not a 30-second epoch (3840 samples) as the first version of this report (wrongly) benchmarked. `generate_embeddings.py` saves this call's per-patch (640 samples = 5s resolution, 60 patches/chunk) embedding stream, and `SleepEventClassificationDataset._try_get_item` (`sleepfm/models/dataset.py`) mean-pools every 6 consecutive patch embeddings ("Resample embeddings from 5-second to 30-second resolution") to build the 30s-epoch sequence the staging head actually consumes -- so the head's shape (960 x 128 embeddings/night) was already correct and is unchanged. BIOT's shape was verified unchanged too: one call per 30s epoch, 200Hz, 6000 samples, `n_fft=200, hop_length=100` (`scripts/biot_dataset.py`, `scripts/finetune_biot.py`).

**Second independent method, and why the two disagree (task item 5):** `fvcore.nn.FlopCountAnalysis` (run with autograd enabled), plus hand-added terms for `nn.LSTM` and `F.scaled_dot_product_attention` (SDPA) -- both opaque fused kernels that NEITHER method decomposes (confirmed directly: both appear in fvcore's own `unsupported_ops()` diagnostic for every SleepFM row). The SDPA correction (`4*batch*seq_len^2*embed_dim` FLOPs per call) was verified exactly against `FlopCounterMode` itself on an isolated `nn.TransformerEncoderLayer` call made WITHOUT `torch.no_grad()` (which forces the slow, decomposed path): expected 471,859,200, measured `aten.bmm`=471,859,200, an exact match. The much bigger, separate reason FlopCounterMode's own numbers are unreliable for this architecture specifically: under `torch.no_grad()` (how we correctly run every real inference benchmark here), PyTorch's `TransformerEncoderLayer`/`MultiheadAttention` take an entirely fused, opaque inference fast-path that FlopCounterMode's op-level dispatch hook cannot see into AT ALL -- verified directly: the real `SleepEventLSTMClassifier` head benchmark call attributes 100% of FlopCounterMode's reported total to its own final `nn.Linear` alone, missing `spatial_pooling`, `transformer_encoder`, AND `lstm` entirely, while the identical module called the identical way but WITHOUT `torch.no_grad()` shows its full, expected `addmm`+`bmm` breakdown. `fvcore` (run with grad enabled, so this fast-path isn't taken) avoids that specific blind spot for the exposed Linear/matmul ops, which is why its raw numbers are so much larger -- but it still separately misses LSTM and SDPA, hence the hand-adds. BIOT's two methods agree closely (within 0.3%) because BIOT's attention is NOT PyTorch's `nn.TransformerEncoderLayer` at all -- it's `linear_attention_transformer`'s linear attention (no quadratic softmax(QK)V, confirmed: `aten::scaled_dot_product_attention` never appears in BIOT's `fvcore_unsupported_ops`), so it never hits either blind spot.


*No marks: FLOPs counts, not a ranked comparison.*
<!-- spec: items=FLOPs counts; marks=none; note=FLOPs counts, not a ranked comparison -->
| Model | Component | Condition | Real input shape | FLOPs (FlopCounterMode) | FLOPs (second method: fvcore + hand-added LSTM/SDPA/STFT/elementwise) | FLOPs / 8h night (second method) |
|---|---|---|---|---|---|---|
| BIOT | end-to-end | 1 channel | (1, 1, 6000) -- 30s epoch @ 200Hz, verified unchanged from Phase 1 | 381,982,720 | 385,330,531 | 366,703,411,200 |
| BIOT | end-to-end | 3 channels | (1, 3, 6000) -- 30s epoch @ 200Hz, verified unchanged from Phase 1 | 1,145,943,040 | 1,159,557,834 | 1,100,105,318,400 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel | (1, 1, 38400) -- 5-MIN CHUNK @ 128Hz, 60 patches of 640 samples each -- CORRECTED from the previous (1,1,3840)/30s-epoch shape, which was wrong | 97,966,080 (per 5-min chunk, unreliable -- see note) | 613,753,440 (per 5-min chunk) / 61,375,344 (per 30s-epoch equivalent) | 58,920,330,240 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels | (1, 3, 38400) -- 5-MIN CHUNK @ 128Hz, 60 patches of 640 samples each -- CORRECTED from the previous (1,3,3840)/30s-epoch shape, which was wrong | 293,898,240 (per 5-min chunk, unreliable -- see note) | 1,035,636,000 (per 5-min chunk) / 103,563,600 (per 30s-epoch equivalent) | 99,421,056,000 |
| SleepFM (From-Scratch/Next-Token share this architecture) | head | n/a (operates on 960x128 embeddings, whole night at once, unchanged shape) | 960x128 embeddings (unchanged) | 1,228,800 (per night, unreliable -- see note) | 7,005,550,080 (per night) | 7,005,550,080 |

**The "39x" statement is fixed (task item 5):** the first version of this report claimed BIOT's end-to-end FLOPs/epoch was a constant ~39x the SleepFM encoder's, at both 1 and 3 channels. That number was an artifact of two compounding errors (the wrong 30s-epoch shape for the encoder, and FlopCounterMode's no-grad blind spot, both explained above) that happened to cancel into a suspiciously exact, channel-count-independent ratio -- in hindsight itself a sign something was wrong, since real attention-based costs don't scale that way. Corrected (second method, FLOPs/8h-night, full scope: BIOT end-to-end vs. SleepFM encoder+head together): BIOT is about **5.6x** the SleepFM encoder+head at 1 channel (369,917,309,978 / 65,925,880,320) and about **10.5x** at 3 channels (1,113,175,520,335 / 106,426,606,080) -- genuinely channel-count-dependent, not constant.

### E3: latency, throughput, and peak GPU memory
Redone again (task items 5-6) on the CORRECTED SleepFM encoder shape (5-minute chunk, `(B, n_ch, 38400)` -- see E2) on every node this was benchmarked on. BIOT is fed its own `(B, 1, 6000)` / `(B, 3, 6000)` per condition (unchanged, verified correct). The staging head is benchmarked separately on a whole night of embeddings (`n_ch=3`, unchanged, verified correct -- see E2's note). Every (model, condition, device) cell is measured 3 independent times, each with its own 10 warm-up and 100 timed calls, CUDA-synchronized around every call; the 3 repeats are summarized below as mean and range (not listed separately -- this is a change from the previous version of this table, which listed all 3 raw values). Excludes data loading and preprocessing -- only the model forward pass is timed. **Labeling every column honestly (task item 6):** every "GPU batch=N" row is the latency of ONE call that processes N inputs together (N in {1, 16, 128}); every "CPU threads=N" row is the latency of ONE call that processes a FIXED batch of 16 inputs using N CPU threads (16 was the batch size used for every CPU row in this benchmark, GPU and CPU batch size are not the same thing and neither varies within its own column). "Per call" here means per 5-minute chunk for the SleepFM encoder and per 30-second epoch for BIOT -- these are different-sized units of work, not directly comparable without multiplying by how many of each a night actually needs (see the calls-per-night table below). GPU latency is flat across batch sizes (1/16/128) for every model/component here -- these are all small models relative to a GH200, so at these batch sizes the benchmark is latency-bound (kernel-launch/memory overhead), not compute-bound; throughput scales with batch size instead (visible in the throughput column). GPU: exclusive GH200 node (`gpumedium`, `--exclusive`, 45-minute limit). CPU: exclusive node on the `medium` CPU partition (`--exclusive`, 45-minute limit) -- the `small` partition rejects `--exclusive` ("administratively prohibited on shared partitions"); `medium` accepts it. The shared-node run (no `--exclusive`, normal `gpumedium`) is kept alongside, labeled by hardware below.

**Shared-node (GPU + CPU together, `gpumedium`, no `--exclusive`):**

*No marks: latency benchmark, not a ranked comparison.*
<!-- spec: items=latency benchmark (shared-node); marks=none; note=latency benchmark, not a ranked comparison -->
| Model | Component | Condition | Device | Median latency (ms, mean of 3 repeats) | p95 latency (ms, mean of 3 repeats) | Throughput (epochs/s, mean of 3 repeats) | Peak GPU memory at batch 128 (MB) |
|---|---|---|---|---|---|---|---|
| BIOT | end-to-end | 1 channel | GPU batch=1 | 2.929 (range 2.920-2.945) | 3.014 (range 2.996-3.044) | 341.4 (range 339.5-342.4) | n/a |
| BIOT | end-to-end | 1 channel | GPU batch=16 | 2.987 (range 2.952-3.039) | 3.649 (range 3.007-4.886) | 5358.1 (range 5265.6-5419.7) | n/a |
| BIOT | end-to-end | 1 channel | GPU batch=128 | 3.070 (range 3.035-3.129) | 4.880 (range 4.822-4.956) | 41707.9 (range 40912.7-42171.1) | 154.285 |
| BIOT | end-to-end | 1 channel | CPU threads=1 | 87.145 (range 87.080-87.193) | 87.603 (range 87.254-88.182) | 183.6 (range 183.5-183.7) | n/a |
| BIOT | end-to-end | 1 channel | CPU threads=8 | 72.053 (range 71.976-72.102) | 72.235 (range 72.196-72.262) | 222.1 (range 221.9-222.3) | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=1 | 3.399 (range 3.353-3.460) | 4.857 (range 3.493-5.568) | 294.2 (range 289.0-298.2) | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=16 | 3.525 (range 3.463-3.599) | 5.027 (range 3.545-5.781) | 4539.7 (range 4445.7-4619.8) | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=128 | 7.006 (range 6.922-7.158) | 7.595 (range 7.583-7.611) | 18273.9 (range 17881.3-18492.6) | 335.178 |
| BIOT | end-to-end | 3 channels | CPU threads=1 | 254.685 (range 254.627-254.747) | 255.011 (range 254.863-255.167) | 62.8 (range 62.8-62.8) | n/a |
| BIOT | end-to-end | 3 channels | CPU threads=8 | 208.750 (range 208.115-209.677) | 209.247 (range 208.469-210.042) | 76.6 (range 76.3-76.9) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (B, 1, 38400) -- 5-min chunk | GPU batch=1 | 2.707 (range 2.698-2.722) | 2.794 (range 2.752-2.873) | 369.4 (range 367.3-370.6) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (B, 1, 38400) -- 5-min chunk | GPU batch=16 | 2.670 (range 2.666-2.675) | 2.724 (range 2.710-2.739) | 5993.0 (range 5980.7-6002.5) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (B, 1, 38400) -- 5-min chunk | GPU batch=128 | 5.185 (range 5.175-5.196) | 5.229 (range 5.227-5.232) | 24687.1 (range 24635.1-24733.4) | 196.055 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (16, 1, 38400) -- 5-min chunk | CPU threads=1 | 177.650 (range 177.520-177.833) | 178.231 (range 177.870-178.896) | 90.1 (range 90.0-90.1) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (16, 1, 38400) -- 5-min chunk | CPU threads=8 | 130.236 (range 130.192-130.307) | 130.579 (range 130.554-130.622) | 122.9 (range 122.8-122.9) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (B, 3, 38400) -- 5-min chunk | GPU batch=1 | 3.090 (range 3.080-3.102) | 3.155 (range 3.146-3.165) | 323.7 (range 322.4-324.7) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (B, 3, 38400) -- 5-min chunk | GPU batch=16 | 3.065 (range 3.055-3.074) | 3.757 (range 3.130-5.006) | 5220.4 (range 5205.3-5237.8) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (B, 3, 38400) -- 5-min chunk | GPU batch=128 | 11.038 (range 11.035-11.042) | 11.104 (range 11.086-11.125) | 11595.9 (range 11591.9-11599.6) | 458.555 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (16, 3, 38400) -- 5-min chunk | CPU threads=1 | 371.548 (range 371.452-371.701) | 372.933 (range 372.633-373.198) | 43.1 (range 43.0-43.1) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (16, 3, 38400) -- 5-min chunk | CPU threads=8 | 235.077 (range 234.980-235.138) | 235.638 (range 235.522-235.861) | 68.1 (range 68.0-68.1) | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=1 | 2.895 (range 2.885-2.908) | 2.944 (range 2.921-2.988) | 345.4 (range 343.9-346.6) | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=16 | 6.396 (range 6.392-6.400) | 6.535 (range 6.532-6.537) | 2501.4 (range 2500.2-2503.1) | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=128 | 34.836 (range 34.693-35.007) | 35.439 (range 35.276-35.584) | 3674.4 (range 3656.4-3689.5) | 4213.954 |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | CPU threads=1 | 2119.091 (range 2118.481-2119.410) | 2130.155 (range 2128.802-2131.686) | 7.6 (range 7.5-7.6) | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | CPU threads=8 | 1372.849 (range 1371.776-1373.551) | 1388.586 (range 1385.401-1391.016) | 11.7 (range 11.6-11.7) | n/a |

Node: `rg3110` (shared-node). CPU: Neoverse-V2, 288 CPUs. GPU: GPU 0: NVIDIA GH200 120GB (UUID: GPU-d06f5fe1-27c7-625a-8a67-94b49eb65c62), driver 615.71.09. `squeue -w <node>`: (unavailable on this node: [Errno 2] No such file or directory: 'squeue') GPU before: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 3 MiB, 97871 MiB`. GPU after: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 8203 MiB, 97871 MiB`.

**Exclusive-node, GPU (`gpumedium --exclusive`, 4x GH200):**

*No marks: latency benchmark, not a ranked comparison.*
<!-- spec: items=latency benchmark (exclusive-node, GPU); marks=none; note=latency benchmark, not a ranked comparison -->
| Model | Component | Condition | Device | Median latency (ms, mean of 3 repeats) | p95 latency (ms, mean of 3 repeats) | Throughput (epochs/s, mean of 3 repeats) | Peak GPU memory at batch 128 (MB) |
|---|---|---|---|---|---|---|---|
| BIOT | end-to-end | 1 channel | GPU batch=1 | 3.870 (range 3.847-3.894) | 3.953 (range 3.922-4.004) | 258.4 (range 256.8-260.0) | n/a |
| BIOT | end-to-end | 1 channel | GPU batch=16 | 3.932 (range 3.918-3.952) | 4.003 (range 3.983-4.037) | 4069.7 (range 4048.1-4084.0) | n/a |
| BIOT | end-to-end | 1 channel | GPU batch=128 | 3.968 (range 3.960-3.977) | 4.036 (range 4.027-4.043) | 32259.4 (range 32182.3-32324.3) | 154.285 |
| BIOT | end-to-end | 3 channels | GPU batch=1 | 4.469 (range 4.459-4.484) | 4.557 (range 4.517-4.614) | 223.8 (range 223.0-224.3) | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=16 | 4.608 (range 4.597-4.617) | 4.701 (range 4.693-4.709) | 3472.0 (range 3465.5-3480.4) | n/a |
| BIOT | end-to-end | 3 channels | GPU batch=128 | 7.233 (range 7.232-7.233) | 7.269 (range 7.268-7.272) | 17697.0 (range 17695.7-17698.0) | 335.178 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (B, 1, 38400) -- 5-min chunk | GPU batch=1 | 3.313 (range 3.292-3.344) | 3.410 (range 3.335-3.541) | 301.9 (range 299.1-303.8) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (B, 1, 38400) -- 5-min chunk | GPU batch=16 | 3.299 (range 3.296-3.301) | 3.359 (range 3.349-3.377) | 4849.5 (range 4846.7-4854.4) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (B, 1, 38400) -- 5-min chunk | GPU batch=128 | 5.212 (range 5.211-5.214) | 5.228 (range 5.225-5.230) | 24556.4 (range 24547.8-24563.1) | 196.055 |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (B, 3, 38400) -- 5-min chunk | GPU batch=1 | 3.739 (range 3.725-3.746) | 4.839 (range 3.781-6.919) | 267.5 (range 267.0-268.4) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (B, 3, 38400) -- 5-min chunk | GPU batch=16 | 3.506 (range 3.499-3.518) | 3.559 (range 3.546-3.572) | 4563.9 (range 4547.8-4572.7) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (B, 3, 38400) -- 5-min chunk | GPU batch=128 | 11.059 (range 11.058-11.061) | 11.072 (range 11.071-11.073) | 11574.6 (range 11572.2-11575.8) | 458.555 |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=1 | 3.073 (range 3.063-3.088) | 3.119 (range 3.096-3.154) | 325.4 (range 323.8-326.5) | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=16 | 6.443 (range 6.437-6.446) | 8.333 (range 6.467-12.055) | 2483.3 (range 2482.1-2485.7) | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | GPU batch=128 | 35.392 (range 35.358-35.449) | 35.918 (range 35.801-35.978) | 3616.6 (range 3610.8-3620.1) | 4270.205 |

Node: `rg3107` (exclusive-node, GPU). CPU: Neoverse-V2, 288 CPUs. GPU: GPU 0: NVIDIA GH200 120GB (UUID: GPU-a3e9f6d6-9e33-83a3-5417-cfef5e15c96a); GPU 1: NVIDIA GH200 120GB (UUID: GPU-56923bc0-5de0-6826-bdf8-c3ac58418465); GPU 2: NVIDIA GH200 120GB (UUID: GPU-ab58baae-dc9c-7188-1b43-bdc73c13ed40); GPU 3: NVIDIA GH200 120GB (UUID: GPU-d3bea9fe-5a06-7e3c-4daf-696f70535bac), driver 615.71.09/615.71.09/615.71.09/615.71.09. `squeue -w <node>`: (unavailable on this node: [Errno 2] No such file or directory: 'squeue') GPU before: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 0 %, 0 %, 7 MiB, 97871 MiB
1, 0 %, 0 %, 3 MiB, 97871 MiB
2, 0 %, 0 %, 4 MiB, 97871 MiB
3, 0 %, 0 %, 5 MiB, 97871 MiB`. GPU after: `index, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
0, 100 %, 41 %, 8209 MiB, 97871 MiB
1, 0 %, 0 %, 3 MiB, 97871 MiB
2, 0 %, 0 %, 4 MiB, 97871 MiB
3, 0 %, 0 %, 5 MiB, 97871 MiB`.

**Exclusive-node, CPU (`medium --exclusive`):**

*No marks: latency benchmark, not a ranked comparison.*
<!-- spec: items=latency benchmark (exclusive-node, CPU); marks=none; note=latency benchmark, not a ranked comparison -->
| Model | Component | Condition | Device | Median latency (ms, mean of 3 repeats) | p95 latency (ms, mean of 3 repeats) | Throughput (epochs/s, mean of 3 repeats) | Peak GPU memory at batch 128 (MB) |
|---|---|---|---|---|---|---|---|
| BIOT | end-to-end | 1 channel | CPU threads=1 | 68.939 (range 68.648-69.232) | 71.520 (range 71.381-71.760) | 232.1 (range 231.1-233.1) | n/a |
| BIOT | end-to-end | 1 channel | CPU threads=8 | 14.447 (range 14.349-14.570) | 14.977 (range 14.747-15.161) | 1107.5 (range 1098.2-1115.0) | n/a |
| BIOT | end-to-end | 3 channels | CPU threads=1 | 218.228 (range 214.757-220.636) | 224.270 (range 222.725-225.679) | 73.3 (range 72.5-74.5) | n/a |
| BIOT | end-to-end | 3 channels | CPU threads=8 | 44.606 (range 43.987-44.948) | 48.685 (range 47.929-49.429) | 358.7 (range 356.0-363.7) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (16, 1, 38400) -- 5-min chunk | CPU threads=1 | 115.059 (range 115.039-115.085) | 115.518 (range 115.468-115.583) | 139.1 (range 139.0-139.1) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 1 channel -- (16, 1, 38400) -- 5-min chunk | CPU threads=8 | 27.084 (range 27.041-27.157) | 27.464 (range 27.423-27.537) | 590.8 (range 589.2-591.7) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (16, 3, 38400) -- 5-min chunk | CPU threads=1 | 228.585 (range 228.416-228.685) | 240.158 (range 238.794-241.522) | 70.0 (range 70.0-70.0) | n/a |
| SleepFM (From-Scratch/Next-Token share this encoder architecture) | encoder | 3 channels -- (16, 3, 38400) -- 5-min chunk | CPU threads=8 | 62.588 (range 62.120-63.056) | 78.471 (range 77.141-80.344) | 255.6 (range 253.7-257.6) | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | CPU threads=1 | 2251.846 (range 2250.156-2254.986) | 2255.910 (range 2253.653-2259.230) | 7.1 (range 7.1-7.1) | n/a |
| SleepFM (From-Scratch/Next-Token share this architecture) | head (whole night, n_ch=3) | n/a (operates on embeddings) | CPU threads=8 | 592.349 (range 591.306-593.033) | 595.345 (range 594.804-595.669) | 27.0 (range 27.0-27.1) | n/a |

Node: `rc5148` (exclusive-node, CPU). CPU: AMD EPYC 9965 192-Core Processor, 384 CPUs. GPU: (unavailable on this node: [Errno 2] No such file or directory: 'nvidia-smi'), driver (unavailable on this node: [Errno 2] No such file or directory: 'nvidia-smi'). `squeue -w <node>`:              JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
           3109981    medium         channel_e3_excl_cpu_v2   hamehd  R       0:06      1 rc5148
 GPU before: `(unavailable on this node: [Errno 2] No such file or directory: 'nvidia-smi')`. GPU after: `(unavailable on this node: [Errno 2] No such file or directory: 'nvidia-smi')`.

`squeue`/co-tenancy could not be queried directly from inside any of these jobs (`squeue` is not on the compute node's PATH); GPU memory/utilization before and after, the node's hostname, and the hardware (CPU model/count, GPU name/driver) captured via `lscpu`/`nvidia-smi` above are the direct signal available for shared-vs-exclusive co-tenancy and cross-node comparability here.

**Seconds per 8h night AND per native call (task item 6), GPU per call of 128 inputs and CPU per call of 16 inputs (8 threads), SleepFM head included -- exclusive-node data:**
Calls/night: BIOT needs 960 (one per 30-second epoch); the SleepFM encoder needs 96 (one per 5-minute chunk -- corrected, see E2); the SleepFM head needs 1 (it runs once per whole night). Seconds/night = calls/night divided by the measured throughput at that batch size (NOT calls/night times the raw per-call latency shown -- that latency is for a BATCHED call of many inputs at once, not a single night in isolation). "SleepFM total" is encoder+head together.

*No marks: derived from measured latency and throughput, not a ranked comparison.*
<!-- spec: items=seconds per native call and per 8h night; marks=none; note=derived from measured latency and throughput, not a ranked comparison -->
| Condition | Device | Calls/night (BIOT) | BIOT ms/call | BIOT s/night | Calls/night (SleepFM encoder) | SleepFM encoder ms/call | SleepFM encoder s/night | SleepFM head s/night | SleepFM total s/night |
|---|---|---|---|---|---|---|---|---|---|
| 1 channel | GPU, batch=128 | 960 | 3.968 | 0.030 | 96 | 5.212 | 0.004 | 0.000 | 0.004 |
| 3 channels | GPU, batch=128 | 960 | 7.233 | 0.054 | 96 | 11.059 | 0.008 | 0.000 | 0.009 |
| 1 channel | CPU, 8 threads, batch=16 | 960 | 14.447 | 0.867 | 96 | 27.084 | 0.163 | 0.037 | 0.200 |
| 3 channels | CPU, 8 threads, batch=16 | 960 | 44.606 | 2.676 | 96 | 62.588 | 0.376 | 0.037 | 0.413 |

Read per call vs. per night separately: at GPU batch=128, BIOT's own call is actually CHEAPER than the SleepFM encoder's (3.968ms vs. 5.212ms at 1 channel; 7.233ms vs. 11.059ms at 3 channels) -- BIOT is NOT more expensive per call here. **BIOT costs more per NIGHT** purely because it needs 10x more calls (960 vs. 96): 0.030s vs. 0.004s at 1 channel, 0.054s vs. 0.008s at 3 channels (GPU); similarly on CPU. This is the opposite of what a "per-call" reading would suggest, which is exactly why the two are reported separately above rather than as a single ambiguous "latency" number.

**CPU figure vs. the earlier (round 2) run, explained (task item 6):** this round's exclusive-CPU BIOT 1-channel, 8-thread figure is 14.447ms; the previous round's was 17.316/16.976/16.966ms (about 17ms) for the SAME code path and the SAME batch size (16) -- BIOT's benchmark was not touched by this round's shape fix (only the SleepFM encoder's input shape changed). The single-thread figures for the SAME comparison match closely (round 2: 69.104/70.353/69.644ms; round 3: 69.232/68.937/68.648ms), which rules out a methodology change or a code bug -- whatever is different is specific to the 8-thread case. The two rounds ran on DIFFERENT physical nodes in the `medium` partition's exclusive-CPU pool: round 2 was `rc4153` (CPU model not captured -- `lscpu` capture was added only this round), round 3 is `rc5148` (AMD EPYC 9965, captured above). Given single-thread performance is essentially identical between the two nodes but 8-thread performance differs by about 17%, the most likely explanation is node-to-node variation in multi-core scaling (NUMA topology, core-to-cache layout, or a different CPU SKU within the same partition) -- not re-measured on a third node, since the single-thread match already rules out the methodology explanations that would matter for this report's conclusions.


### E4: input data size per channel per night

*No marks: input size, not a ranked comparison.*
<!-- spec: items=input data size; marks=none; note=input size, not a ranked comparison -->
| Model | Sampling rate (Hz) | Samples / channel / night | Bytes / channel / night | MB / channel / night |
|---|---|---|---|---|
| SleepFM From-Scratch | 128 | 3,686,400 | 14,745,600 | 14.062 |
| SleepFM Next-Token | 128 | 3,686,400 | 14,745,600 | 14.062 |
| BIOT | 200 | 5,760,000 | 23,040,000 | 21.973 |


### E5: training cost from sacct
Unchanged from the previous revision of this file (not in scope for this task's items). `sacct -S 2026-07-01 -u hamehdi` (the pre-migration account) returns full historical accounting data -- the earlier "not recoverable" conclusion was querying the wrong account. Every job id below was matched to its (model, condition, fold) by its own `SubmitLine` (sacct), which names the exact config file and timestamp -- unambiguous. Epoch counts for most fine-tuning runs came from a `.log` file either recovered from `/projappl/project_2019517/archive/logs_2026-11-11.tar.gz` (extracted to a temp dir, deleted after use) or, for both pretraining runs, the `epoch` field inside the actual encoder checkpoint, cross-checked against a per-epoch training log whose filename timestamp matches the job's start time. Reproducible via `scripts/analysis/training_cost.py`.


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
Fixed again (task item 7): now compares BIOT against each SleepFM model at the SAME condition on both sides (EEG3 only against EEG3 only; 3-channel benchmark against 3-channel benchmark -- never EEG3-only against a 3-channel run, or one model's 3-channel benchmark against the other's). Scope: SleepFM = encoder + head together (amortized per-night -- see E2); BIOT = end-to-end. The macro F1 difference is the PRIMARY statistic's cross-model paired pooled bootstrap (same 2056 subjects, same resampled indices for both models each draw -- same method as D.2's within-model primary comparisons, just between models instead of between conditions). Patience is stated for both sides of every row, since it is never the same for the two models' 3-channel runs.

*No marks: head-to-head comparisons, not a ranked comparison.*
<!-- spec: items=matched-condition model comparisons; marks=none; note=head-to-head comparisons, not a ranked comparison -->
| Condition | Model A | Patience A | Model B | Patience B | n subjects | Pooled macro F1 diff (A-B), 95% CI | Compute ratio per night (A/B, FLOPs, second method) |
|---|---|---|---|---|---|---|---|
| EEG3 only | BIOT | patience 10 | SleepFM From-Scratch | patience 10 | 2056 | +0.0110 [+0.0078, +0.0141] | 5.61x (369,917,309,978 / 65,925,880,320) |
| 3-channel benchmark | BIOT | patience 20 | SleepFM From-Scratch | patience 50 | 2056 | +0.0088 [+0.0059, +0.0115] | 10.46x (1,113,175,520,335 / 106,426,606,080) |
| EEG3 only | BIOT | patience 10 | SleepFM Next-Token | patience 10 | 2056 | +0.0034 [+0.0002, +0.0065] | 5.61x (369,917,309,978 / 65,925,880,320) |
| 3-channel benchmark | BIOT | patience 20 | SleepFM Next-Token | patience 50 | 2056 | +0.0144 [+0.0113, +0.0171] | 10.46x (1,113,175,520,335 / 106,426,606,080) |

Diff = Model A minus Model B; a positive diff means BIOT scores higher. The compute ratio does not depend on which SleepFM model it's compared to (both share the same encoder architecture and FLOPs), only on the condition (1 channel vs. 3 channels) -- it is shown once per row for direct reading next to that row's macro F1 difference, not because it changes. At EEG3 only, BIOT scores 0.003-0.011 macro F1 higher for 5.6x the compute; at the 3-channel benchmark, BIOT scores 0.009-0.014 higher for 10.5x the compute, but BIOT's 3-channel run also used a lower patience (20 vs. 50) than SleepFM's, so part of that gap is not a clean architecture comparison -- see D.6 for the full patience/pretraining caveats.

## GPU-hours used
Covers every job run while producing this report (Phase 1 and both follow-up rounds), via `sacct -u hamehd` (excluding `.batch`/`.extern` sub-steps). Part A, C, D (all of it -- the confusion-matrix build and the bootstrap statistics both ran on a login-node CPU venv), E1, and E4 needed no GPU, so they do not appear here. This is this task's OWN compute cost, distinct from Part E5's historical training-cost ledger for the runs already in `results/`.

*No marks: job list, GPU-hours summed below.*
<!-- spec: items=job list; marks=none; note=job list, GPU-hours summed below -->
| Job ID | Purpose | Elapsed | GPUs | State |
|---|---|---|---|---|
| 3084148 | Phase 1: BIOT eval rerun (first attempt, pre-incident) | 01:15:05 | 1 | completed |
| 3092867 | Phase 1: BIOT eval rerun (clean, verified, --no_checkpoint_link) | 01:10:45 | 1 | completed |
| 3093075 | Phase 1: Part E2 FLOPs (first attempt) | 00:00:10 | 1 | completed |
| 3093089 | Phase 1: Part E3 exclusive-node probe (returned in ~3s) | 00:00:03 | 4 | completed |
| 3093108 | Phase 1: Part E3 shared-node (first attempt, superseded) | 00:00:07 | 1 | completed |
| 3093150 | Phase 1: Part E3 shared-node (superseded) | 00:01:15 | 1 | completed |
| 3093319 | Phase 1: Part E2 FLOPs (canonical run, superseded this round) | 00:00:10 | 1 | completed |
| 3094289 | Phase 1: Part E3 exclusive-node (superseded) | 00:01:17 | 4 | completed |
| 3105302 | Round 2: E3 shared-node v2, attempt 1 -- crashed, missing `module load python-pytorch/2.10` | 00:00:02 | 1 | FAILED |
| 3105303 | Round 2: E3 exclusive-GPU v2, attempt 1 -- same missing-module crash | 00:00:02 | 4 | FAILED |
| 3105315 | Round 2: E3 shared-node v2, attempt 2 -- ran CPU-only (CUDA silently unavailable, same missing module), cancelled once diagnosed | 00:06:11 | 1 | cancelled |
| 3105316 | Round 2: E3 exclusive-GPU v2, attempt 2 -- completed but produced 0 rows (same missing module) | 00:00:10 | 4 | completed, unused |
| 3105366 | Round 2: E3 exclusive-GPU v2, attempt 3 -- cancelled while still queued, once the fix was ready | 00:00:00 | 0 | cancelled |
| 3105306 | Round 2: E3 exclusive-CPU v2 (superseded this round -- wrong SleepFM encoder shape) | 00:18:00 | 0 | completed, superseded |
| 3105424 | Round 2: E3 shared-node v2 (superseded this round -- wrong SleepFM encoder shape) | 00:23:45 | 1 | completed, superseded |
| 3105425 | Round 2: E3 exclusive-GPU v2 (superseded this round -- wrong SleepFM encoder shape) | 00:00:46 | 4 | completed, superseded |
| 3109762 | Round 3: probe node hardware + fvcore availability | 00:00:09 | 1 | completed |
| 3109808 | Round 3: Part E2 FLOPs v2, first pass (SDPA hand-add not yet added) | 00:00:10 | 1 | completed, superseded |
| 3109850 | Round 3: FLOPs diagnostic (isolated SDPA/TransformerEncoderLayer probe) | 00:00:08 | 1 | completed |
| 3109909 | Round 3: Part E2 FLOPs v2, final (used above) | 00:00:09 | 1 | completed |
| 3109980 | Round 3: E3 exclusive-GPU v3, corrected 5-min-chunk shape (used above) | 00:00:51 | 4 | completed |
| 3109981 | Round 3: E3 exclusive-CPU v3, corrected 5-min-chunk shape (used above) | 00:20:08 | 0 | completed |
| 3109982 | Round 3: E3 shared-node v3, attempt 1 -- TIMEOUT at the 30-minute limit (10x bigger CPU-side input than round 2 took longer than expected; no partial output, script only writes JSON at the end) | 00:30:29 | 1 | TIMEOUT |
| 3131554 | Round 3: E3 shared-node v3, attempt 2, 75-minute limit (used above) | 00:28:27 | 1 | completed |

Total across every row above: **4.16 GPU-hours**. Round 2 includes three failed/superseded attempts caused by a missing `module load python-pytorch/2.10` line; round 2's E3 benchmark rows are themselves superseded this round, once the SleepFM encoder's real 5-minute-chunk input shape was verified (see E2's note) and the benchmark was corrected and rerun.

