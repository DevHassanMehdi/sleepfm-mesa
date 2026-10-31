# Single-EEG-Channel Comparison: Summary

Full details, per-fold numbers, and provenance are in
`results/detailed/EEG_CHANNEL_DETAILED.md`. This file summarizes that
data; every number below is taken from it.

## Configuration

| Setting | Value |
|---|---|
| Cohort | 2056 subjects (full MESA cohort) |
| Fold scheme | `fold5_v1`, 5 folds, disjoint test sets (411/411/411/411/412) |
| Single-channel fine-tuning patience | 10, for all nine single-channel runs (3 models × EEG1/EEG2/EEG3) |
| 3-channel reference patience | From-Scratch: 10 (rerun). Next-Token: 10 (rerun). BIOT: 20 (original benchmark) |
| EEG1 | Fz(+) − Cz(−) |
| EEG2 | Cz(+) − Oz(−) |
| EEG3 | C4(+) − M1(−) |

The Fz/Oz/C4 labels used in LaBraM's channel mapping name only the
active electrode and drop the reference -- a simplification, not a
claim that, e.g., EEG1 *is* Fz referenced to a generic ground. All three
are bipolar derivations.

## Table 1: Macro F1 (mean ± std), by model and channel

| Model | EEG1 | EEG2 | EEG3 | 3-channel reference |
|---|---|---|---|---|
| From-Scratch | 0.6835 ± 0.0120 | 0.6843 ± 0.0100 | 0.6886 ± 0.0121 | **0.7306 ± 0.0102** (patience 10) |
| Next-Token | 0.7015 ± 0.0063 | 0.6966 ± 0.0098 | 0.6964 ± 0.0091 | **0.7346 ± 0.0053** (patience 10) |
| BIOT | 0.6976 ± 0.0052 | 0.6887 ± 0.0088 | 0.6998 ± 0.0088 | **0.7541 ± 0.0047** (patience 20) |

Differences below about 0.01 are within the fold-to-fold spread.

## Table 2: Per-class F1, by model and channel

| Model | Channel | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| From-Scratch | EEG1 | 0.9176 | 0.3200 | 0.7826 | 0.6291 | 0.7681 |
| From-Scratch | EEG2 | 0.9226 | 0.3526 | 0.7694 | 0.6231 | 0.7536 |
| From-Scratch | EEG3 | 0.9169 | 0.3394 | 0.7746 | 0.6456 | 0.7663 |
| From-Scratch | 3ch reference (p10) | 0.9431 | 0.4270 | 0.8036 | 0.6583 | 0.8211 |
| Next-Token | EEG1 | 0.9218 | 0.3831 | 0.7901 | 0.6353 | 0.7769 |
| Next-Token | EEG2 | 0.9274 | 0.3912 | 0.7787 | 0.6323 | 0.7534 |
| Next-Token | EEG3 | 0.9213 | 0.3412 | 0.7857 | 0.6501 | 0.7835 |
| Next-Token | 3ch reference (p10) | 0.9425 | 0.4220 | 0.8148 | 0.6708 | 0.8229 |
| BIOT | EEG1 | 0.9070 | 0.4824 | 0.7567 | 0.6244 | 0.7175 |
| BIOT | EEG2 | 0.9149 | 0.4829 | 0.7437 | 0.6174 | 0.6845 |
| BIOT | EEG3 | 0.9181 | 0.4800 | 0.7462 | 0.6396 | 0.7152 |
| BIOT | 3ch reference (p20) | 0.9455 | 0.5363 | 0.8023 | 0.6798 | 0.8068 |

## Discussion

We tested each EEG channel on its own: EEG1 (Fz-Cz), EEG2 (Cz-Oz), and
EEG3 (C4-M1). For From-Scratch and Next-Token, the three channels score
the same within noise, between 0.68 and 0.70. For BIOT, EEG2 scores
lowest, about 0.01 below the others, and it loses on all five folds. The
other two models do not show this. A single channel scores 3 to 5 points
below all three together, so more channels still help. For a wearable,
the choice of channel matters little for overall macro F1. Each channel
was run once per model and the gaps are small.
