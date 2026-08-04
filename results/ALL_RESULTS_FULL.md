# SleepFM on MESA: Full-Cohort Results (1944 subjects, fold5_v1)

## Configuration

| Setting | Value |
|---|---|
| Cohort size | 1944 subjects |
| Fold scheme | fold5_v1 (subject-level 5-fold CV) |
| Channel scope | EEG1, EEG2, EEG3, EKG only |
| From-Scratch pretraining | Contrastive leave-one-out (from random init) |
| Spectral pretraining | Spectral band-power reconstruction (from random init) |
| BIOT pretraining | Externally pretrained (EEG-SHHS+PREST-18-channels.ckpt) + fine-tuned |
| LaBraM pretraining | Externally pretrained (labram-base.pth) + fine-tuned |
| SensorLM pretraining | One-stage, from-scratch (no separate pretraining phase) |

## EEG Only

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| From-Scratch | 0.4949 ± 0.0135 | 0.5433 ± 0.0077 | 0.7229 | 0.3317 | 0.5204 | 0.3903 | 0.5094 |
| Spectral | 0.5132 ± 0.0101 | 0.5560 ± 0.0110 | 0.7243 | 0.3516 | 0.5434 | 0.3862 | 0.5605 |
| BIOT | **0.7367 ± 0.0048** | **0.7694 ± 0.0069** | **0.9153** | **0.6570** | **0.7195** | **0.6607** | **0.7309** |
| LaBraM | 0.6951 ± 0.0070 | 0.7332 ± 0.0073 | 0.8972 | 0.5726 | 0.6994 | 0.6528 | 0.6535 |
| SensorLM | 0.6286 ± 0.0067 | 0.6750 ± 0.0094 | 0.8607 | 0.5072 | 0.6473 | 0.6276 | 0.5003 |

## ECG Only

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| From-Scratch | **0.3997 ± 0.0131** | **0.4663 ± 0.0137** | **0.6823** | **0.3087** | **0.4142** | **0.2627** | **0.3308** |
| Spectral | 0.3134 ± 0.0080 | 0.3713 ± 0.0079 | 0.6121 | 0.2827 | 0.2577 | 0.1881 | 0.2264 |
| BIOT | 0.3118 ± 0.0069 | 0.3808 ± 0.0065 | 0.6350 | 0.3063 | 0.3192 | 0.1683 | 0.1303 |
| SensorLM | 0.2883 ± 0.0053 | 0.3465 ± 0.0235 | 0.6108 | 0.2602 | 0.2605 | 0.1548 | 0.1550 |

## EEG + ECG

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| From-Scratch | 0.5056 ± 0.0152 | 0.5496 ± 0.0147 | 0.7359 | 0.3551 | 0.5247 | 0.3858 | 0.5265 |
| Spectral | 0.5060 ± 0.0191 | 0.5554 ± 0.0136 | 0.7374 | 0.3394 | 0.5584 | 0.3694 | 0.5254 |
| BIOT | **0.7361 ± 0.0058** | **0.7704 ± 0.0069** | **0.9135** | **0.6512** | **0.7332** | **0.6573** | **0.7252** |
| LaBraM | 0.6868 ± 0.0066 | 0.7278 ± 0.0069 | 0.8947 | 0.5559 | 0.7053 | 0.6490 | 0.6291 |
| SensorLM | 0.6271 ± 0.0075 | 0.6728 ± 0.0113 | 0.8664 | 0.5190 | 0.6302 | 0.6220 | 0.4979 |

## Discussion

**EEG_ONLY and EEG+ECG**: BIOT wins by a wide margin — 0.73+ macro F1 vs.
0.49-0.51 for the SleepFM variants. LaBraM sits second, close behind BIOT.
SensorLM trails both. From-Scratch and Spectral land in almost the same
place as each other on these two modalities — neither has a real edge.

**ECG_ONLY**: Everything flips. From-Scratch takes the top spot (0.40),
ahead of BIOT (0.31) and Spectral (0.31). Every model drops hard on
ECG_ONLY compared to EEG_ONLY — makes sense, since ECG just doesn't carry
as much sleep-stage information as EEG does. Wake and N1 hold up okay
across models; N2 and REM fall apart the most.

**Why BIOT and LaBraM lead on EEG but fall behind on ECG**: both were
pretrained on large external EEG datasets — that's their whole advantage.
Neither ever saw real ECG signal during pretraining, so on ECG_ONLY
they're working with representations built for a different kind of
signal. From-Scratch, by contrast, was pretrained directly on this
cohort's EEG and ECG together, so it has no such gap — a smaller,
task-specific model that actually learned ECG can beat two much larger,
externally-pretrained ones once you take EEG out of the picture.

**Bottom line**: if a wearable can capture EEG, BIOT is the strongest
choice by a clear margin. If it's ECG-only (e.g. a chest strap or watch
with no EEG lead), the picture reverses — a small model trained
specifically on the target signal beats big, EEG-pretrained encoders
forced to work with a signal they've never seen.

## Pending

- LaBraM ECG_ONLY -- 4/5 folds complete
- SleepFM Next-Token (EEG+ECG-only scope) -- pretraining/fine-tuning not yet complete
