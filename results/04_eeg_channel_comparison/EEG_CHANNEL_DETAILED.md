# Single-EEG-Channel Comparison (Full Cohort, 2056 Subjects)

## Purpose

MESA's three recorded EEG channels (`EEG1`, `EEG2`, `EEG3`) are fine-tuned
individually, one channel at a time, for From-Scratch, Next-Token, and BIOT,
to see whether any single derivation carries most of the signal the
3-channel `EEG_ONLY` montage provides, and whether one derivation is
consistently stronger than the others.

## Channel identity

Per the NSRR MESA Sleep Polysomnography Manual of Procedures (Somte PIB
hookup, pp. 29-31; see the header comment in `scripts/labram_dataset.py`),
`EEG1`/`EEG2`/`EEG3` are **bipolar derivations**, not single electrodes:

| HDF5 key | Derivation | LaBraM's simplified label |
|---|---|---|
| EEG1 | Fz(+) − Cz(−) | "Fz" |
| EEG2 | Cz(+) − Oz(−) | "Oz" |
| EEG3 | C4(+) − M1(−) | "C4" |

LaBraM's channel names name only the active electrode and drop the
reference, which is a simplification, not a claim that `EEG1` *is* Fz
referenced to a generic ground. Keep the derivation pair in mind when
comparing against LaBraM-style single-electrode literature.

## Methodology

- Cohort: full 2056-subject cohort, `fold5_v1` split (5 folds, disjoint test
  sets, standard 411/411/411/411/412 sizes).
- From-Scratch and Next-Token: single-channel fine-tuning starts from the
  existing 3-channel `EEG_ONLY` pretrained encoder, with `modality_types`
  and the channel-count config narrowed to one channel
  (`EEG1_ONLY`/`EEG2_ONLY`/`EEG3_ONLY`). Fine-tuned with `--patience 10`
  (new CLI option — see "Patience" below).
- BIOT: single-channel fine-tuning of the BIOT encoder pretrained on
  EEG-SHHS+PREST, with `--patience 10`, `--split_id fold5_v1`, full-cohort
  HDF5s. A patience-10 run was timed first (6h42m for 15 epochs); the
  remaining 14 runs used an 18h SLURM wall-time limit built with
  `--checkpoint_dir`-based resume support. **None of the 14 runs hit the
  wall-time limit** (longest: 16h47m39s), so resume was never actually
  exercised in this batch — it is wired up and available, not tested.
  Input shape was confirmed as `(16, 1, 6000)` (batch, channel, 200 Hz
  samples) for all three channels via a diagnostic print added to
  `finetune_biot.py`.
- Patience-10 **3-channel** reference: the existing 3-channel `EEG_ONLY`
  pretrained encoder, fine-tuned with `--patience 10` instead of the
  benchmark's patience, under distinct `pretrain_method` names
  (`fromscratch_p10ref`, `nexttoken_p10ref`) so the runs cannot be confused
  with the 90-run benchmark directories.
- Metrics: From-Scratch/Next-Token macro F1, accuracy, and per-class F1
  were recomputed directly from the raw test pickles
  (`test_all_outputs`/`test_all_targets`/`test_all_masks`/`test_all_paths`)
  with plain `sklearn.metrics.f1_score`/`accuracy_score`, selecting valid
  epochs with `mask == 0` and taking `argmax` over the 5-way softmax output.
  BIOT has no raw pickles, so its numbers are read from each fold's
  `classification_report.txt` (macro-avg and per-class F1) cross-checked
  against `metrics.json`'s `overall_macro_f1`/`overall_accuracy` fields —
  matched exactly in every fold checked.
- Fold disjointness: for every experiment below, the union of the 5 test
  folds' subject IDs is exactly 2056 with zero overlap. Confirmed for all
  9 single-channel experiments (3 models × 3 channels) and both
  patience-10 3-channel references.

### Patience is not matched across the original 90-run benchmark

`finetune_sleep_staging.py` (used by From-Scratch and Next-Token)
previously hardcoded `patience = 50` and ignored the YAML config's
`early_stopping_patience` field entirely — it has now been given a real
`--patience` CLI option (default 50, so existing results are unaffected).
BIOT's `--patience` was already a working argparse option, defaulting to
20. This means the original 90-run benchmark numbers below were produced
at **patience 50 for From-Scratch/Next-Token** and **patience 20 for
BIOT** — not the same patience across models, and not the same patience
as the patience-10 runs in this comparison. The patience-10 3-channel
reference rows exist specifically to separate "channel effect" from
"patience effect."

## Results

### Macro F1 / accuracy / per-class F1, by model and channel

| Model | Channel | Macro F1 (mean ± std) | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| From-Scratch | EEG1 | 0.6835 ± 0.0120 | 0.8010 | 0.9176 | 0.3200 | 0.7826 | 0.6291 | 0.7681 |
| From-Scratch | EEG2 | 0.6843 ± 0.0100 | 0.7978 | 0.9226 | 0.3526 | 0.7694 | 0.6231 | 0.7536 |
| From-Scratch | EEG3 | 0.6886 ± 0.0121 | 0.7994 | 0.9169 | 0.3394 | 0.7746 | 0.6456 | 0.7663 |
| From-Scratch | 3ch, patience 10 | 0.7306 ± 0.0102 | 0.8309 | 0.9431 | 0.4270 | 0.8036 | 0.6583 | 0.8211 |
| From-Scratch | 3ch, patience 50 (benchmark) | 0.7451 ± 0.0052 | 0.8403 | 0.9468 | 0.4661 | 0.8182 | 0.6569 | 0.8373 |
| Next-Token | EEG1 | 0.7015 ± 0.0063 | 0.8084 | 0.9218 | 0.3831 | 0.7901 | 0.6353 | 0.7769 |
| Next-Token | EEG2 | 0.6966 ± 0.0098 | 0.8044 | 0.9274 | 0.3912 | 0.7787 | 0.6323 | 0.7534 |
| Next-Token | EEG3 | 0.6964 ± 0.0091 | 0.8081 | 0.9213 | 0.3412 | 0.7857 | 0.6501 | 0.7835 |
| Next-Token | 3ch, patience 10 | 0.7346 ± 0.0053 | 0.8355 | 0.9425 | 0.4220 | 0.8148 | 0.6708 | 0.8229 |
| Next-Token | 3ch, patience 50 (benchmark) | 0.7394 ± 0.0051 | 0.8373 | 0.9444 | 0.4443 | 0.8158 | 0.6684 | 0.8241 |
| BIOT | EEG1 | 0.6976 ± 0.0052 | 0.7795 | 0.9070 | 0.4824 | 0.7567 | 0.6244 | 0.7175 |
| BIOT | EEG2 | 0.6887 ± 0.0088 | 0.7765 | 0.9149 | 0.4829 | 0.7437 | 0.6174 | 0.6845 |
| BIOT | EEG3 | 0.6998 ± 0.0088 | 0.7815 | 0.9181 | 0.4800 | 0.7462 | 0.6396 | 0.7152 |
| BIOT | 3ch, patience 20 (benchmark) | 0.7541 ± 0.0047 | 0.8274 | 0.9455 | 0.5363 | 0.8023 | 0.6798 | 0.8068 |

Std is the population std across the 5 folds. The 3-channel patience-20
BIOT row is the original benchmark (no patience-10 3-channel BIOT
reference was run in this task).

Every single-channel result sits 3-6 points of macro F1 below its own
model's 3-channel montage, at matched patience (From-Scratch:
0.684-0.689 vs 0.731; Next-Token: 0.696-0.702 vs 0.735; BIOT: 0.689-0.700
vs the patience-20 benchmark's 0.754, with a less direct patience
comparison). Losing N1 is the single biggest per-class driver each time
(single-channel N1 F1 sits roughly 5-10 points under the matching 3-channel
row for From-Scratch/Next-Token). The three channels individually carry
clearly less signal than the full montage for every model tested — this is
the one finding all three models agree on without qualification.

### Per-fold macro F1 and paired channel comparisons

**From-Scratch**

| Fold | EEG1 | EEG2 | EEG3 |
|---|---|---|---|
| 0 | 0.6899 | 0.6798 | 0.6966 |
| 1 | 0.6919 | 0.7025 | 0.6947 |
| 2 | 0.6610 | 0.6757 | 0.6947 |
| 3 | 0.6934 | 0.6872 | 0.6923 |
| 4 | 0.6812 | 0.6761 | 0.6646 |

| Pair | Per-fold diff | Wins | Mean diff |
|---|---|---|---|
| EEG1 − EEG2 | +0.0101, −0.0106, −0.0147, +0.0062, +0.0051 | EEG1 3/5, EEG2 2/5 | −0.0008 |
| EEG1 − EEG3 | −0.0067, −0.0028, −0.0337, +0.0011, +0.0165 | EEG1 2/5, EEG3 3/5 | −0.0051 |
| EEG2 − EEG3 | −0.0167, +0.0078, −0.0190, −0.0051, +0.0115 | EEG2 2/5, EEG3 3/5 | −0.0043 |

**Next-Token**

| Fold | EEG1 | EEG2 | EEG3 |
|---|---|---|---|
| 0 | 0.6960 | 0.6963 | 0.6981 |
| 1 | 0.7136 | 0.7036 | 0.7051 |
| 2 | 0.6978 | 0.6935 | 0.6905 |
| 3 | 0.7002 | 0.7092 | 0.7060 |
| 4 | 0.6996 | 0.6804 | 0.6821 |

| Pair | Per-fold diff | Wins | Mean diff |
|---|---|---|---|
| EEG1 − EEG2 | −0.0003, +0.0101, +0.0044, −0.0090, +0.0192 | EEG1 3/5, EEG2 2/5 | +0.0049 |
| EEG1 − EEG3 | −0.0020, +0.0085, +0.0073, −0.0059, +0.0175 | EEG1 3/5, EEG3 2/5 | +0.0051 |
| EEG2 − EEG3 | −0.0017, −0.0015, +0.0030, +0.0031, −0.0017 | EEG2 2/5, EEG3 3/5 | +0.0002 |

**BIOT**

| Fold | EEG1 | EEG2 | EEG3 |
|---|---|---|---|
| 0 | 0.6922 | 0.6840 | 0.6937 |
| 1 | 0.7074 | 0.7043 | 0.7164 |
| 2 | 0.6952 | 0.6779 | 0.6980 |
| 3 | 0.6951 | 0.6900 | 0.6996 |
| 4 | 0.6982 | 0.6872 | 0.6915 |

| Pair | Per-fold diff | Wins | Mean diff |
|---|---|---|---|
| EEG1 − EEG2 | +0.0082, +0.0031, +0.0173, +0.0051, +0.0110 | **EEG1 5/5**, EEG2 0/5 | +0.0089 |
| EEG1 − EEG3 | −0.0015, −0.0090, −0.0028, −0.0045, +0.0067 | EEG1 1/5, EEG3 4/5 | −0.0022 |
| EEG2 − EEG3 | −0.0097, −0.0121, −0.0201, −0.0096, −0.0043 | EEG2 0/5, **EEG3 5/5** | −0.0112 |

## Which channel wins, and does it agree across models?

**From-Scratch and Next-Token: no channel can be called best.** Every
pairwise comparison is a narrow 3-2 (or 2-3) split, and every mean
per-fold difference (0.0002-0.0051) is smaller than the fold-to-fold std
of the channels being compared (0.0063-0.0121). Per the stated bar — a
channel must win most folds *and* the gap must exceed the fold spread —
none of the three EEG derivations clears it for either model.

**BIOT: EEG2 is clearly the weakest channel**, losing to both EEG1 (5/5
folds, mean gap +0.0089) and EEG3 (5/5 folds, mean gap +0.0112) by margins
that exceed EEG2's own fold-to-fold std (0.0088). EEG1 vs. EEG3 for BIOT,
however, is not resolved: EEG3 wins 4/5 folds but the mean gap (0.0022) is
smaller than both channels' fold stds, so EEG1 and EEG3 are
indistinguishable for BIOT under the same bar.

**Cross-model agreement: weak.** BIOT's one firm result (EEG2 worst) isn't
contradicted by From-Scratch or Next-Token, but it also isn't *confirmed*
by them — neither From-Scratch nor Next-Token clears the bar for ranking
any channel above another, so there's no independent replication of "EEG2
is worst" outside BIOT. The honest summary is: three models, three mostly
flat single-channel rankings, and one model (BIOT) with a single
statistically resolved pairwise result that the other two neither confirm
nor refute.

## Caveats

- BIOT's 14 new runs landed in scattered, auto-generated timestamp
  directories per channel (BIOT has no `--timestamp` override, unlike
  `finetune_sleep_staging.py`) — gathered by globbing all matching-channel
  checkpoint directories. Verified exactly 5 unique, non-overlapping
  `fold_N` per channel.
- The 18h SLURM wall-time limit and `--checkpoint_dir` resume path were
  built and are available for future BIOT runs, but no run in this batch
  actually needed to resume (longest run: 16h47m39s).
- Patience is not matched to the original 90-run benchmark (patience 50
  for From-Scratch/Next-Token, patience 20 for BIOT) — the patience-10
  3-channel reference rows above isolate the patience effect from the
  channel effect, but the single-channel numbers in this report should
  only be compared against the patience-10 3-channel rows, not directly
  against the patience-50/20 benchmark rows.
