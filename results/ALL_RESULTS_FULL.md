# SleepFM on MESA: Full-Cohort Results (1944 subjects, fold5_v1)

## Configuration

| Setting | Value |
|---|---|
| Cohort size | 1944 subjects |
| Fold scheme | fold5_v1 (subject-level 5-fold CV) |
| Channel scope | EEG1, EEG2, EEG3, EKG only |
| From-Scratch pretraining | Contrastive leave-one-out (from random init) |
| Spectral pretraining | Spectral band-power reconstruction (from random init) |
| Next-Token pretraining | Next-window token prediction, 512-cluster codebook (from random init) |
| BIOT pretraining | Externally pretrained (EEG-SHHS+PREST-18-channels.ckpt) + fine-tuned |
| LaBraM pretraining | Externally pretrained (labram-base.pth) + fine-tuned |
| SensorLM pretraining | One-stage, from-scratch (no separate pretraining phase) |

## EEG Only

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| From-Scratch | 0.4949 ± 0.0135 | 0.5433 ± 0.0077 | 0.7229 | 0.3317 | 0.5204 | 0.3903 | 0.5094 |
| Spectral | 0.5132 ± 0.0101 | 0.5560 ± 0.0110 | 0.7243 | 0.3516 | 0.5434 | 0.3862 | 0.5605 |
| Next-Token | 0.5238 ± 0.0087 | 0.5711 ± 0.0109 | 0.7314 | 0.3490 | 0.5683 | 0.3868 | 0.5837 |
| BIOT | **0.7367 ± 0.0048** | **0.7694 ± 0.0069** | **0.9153** | **0.6570** | **0.7195** | **0.6607** | **0.7309** |
| LaBraM | 0.6951 ± 0.0070 | 0.7332 ± 0.0073 | 0.8972 | 0.5726 | 0.6994 | 0.6528 | 0.6535 |
| SensorLM | 0.6286 ± 0.0067 | 0.6750 ± 0.0094 | 0.8607 | 0.5072 | 0.6473 | 0.6276 | 0.5003 |

## ECG Only

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| From-Scratch | **0.3997 ± 0.0131** | **0.4663 ± 0.0137** | **0.6823** | **0.3087** | **0.4142** | **0.2627** | **0.3308** |
| Spectral | 0.3134 ± 0.0080 | 0.3713 ± 0.0079 | 0.6121 | 0.2827 | 0.2577 | 0.1881 | 0.2264 |
| Next-Token | 0.3418 ± 0.0112 | 0.4071 ± 0.0081 | 0.6627 | 0.2623 | 0.3233 | 0.2185 | 0.2421 |
| BIOT | 0.3118 ± 0.0069 | 0.3808 ± 0.0065 | 0.6350 | 0.3063 | 0.3192 | 0.1683 | 0.1303 |
| LaBraM | 0.2971 ± 0.0111 | 0.3802 ± 0.0237 | 0.6133 | 0.2522 | 0.3611 | 0.1260 | 0.1328 |
| SensorLM | 0.2883 ± 0.0053 | 0.3465 ± 0.0235 | 0.6108 | 0.2602 | 0.2605 | 0.1548 | 0.1550 |

## EEG + ECG

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| From-Scratch | 0.5056 ± 0.0152 | 0.5496 ± 0.0147 | 0.7359 | 0.3551 | 0.5247 | 0.3858 | 0.5265 |
| Spectral | 0.5060 ± 0.0191 | 0.5554 ± 0.0136 | 0.7374 | 0.3394 | 0.5584 | 0.3694 | 0.5254 |
| Next-Token | 0.5223 ± 0.0158 | 0.5705 ± 0.0138 | 0.7376 | 0.3602 | 0.5667 | 0.3886 | 0.5585 |
| BIOT | **0.7361 ± 0.0058** | **0.7704 ± 0.0069** | **0.9135** | **0.6512** | **0.7332** | **0.6573** | **0.7252** |
| LaBraM | 0.6868 ± 0.0066 | 0.7278 ± 0.0069 | 0.8947 | 0.5559 | 0.7053 | 0.6490 | 0.6291 |
| SensorLM | 0.6271 ± 0.0075 | 0.6728 ± 0.0113 | 0.8664 | 0.5190 | 0.6302 | 0.6220 | 0.4979 |

## Discussion

**Across all three SleepFM variants** (From-Scratch, Spectral, Next-Token),
the differences are small — all three land close together on every
modality, with Next-Token consistently a touch ahead of the other two on
EEG_ONLY and ECG_ONLY, and From-Scratch still holding the top ECG_ONLY
spot overall. None of the three pretraining objectives (contrastive,
spectral reconstruction, next-window prediction) produces a dramatically
better encoder than the others for this task — they're variations on a
theme, not different tiers.

**BIOT and LaBraM dominate EEG-based modalities** (EEG_ONLY, EEG_ECG) by a
wide margin over every SleepFM variant and SensorLM — both were pretrained
on large external EEG datasets, and that head start shows. BIOT leads
outright; LaBraM sits close behind.

**ECG_ONLY flips the ranking entirely.** From-Scratch takes the top spot,
followed by Next-Token — both SleepFM variants, both pretrained directly
on this cohort's actual ECG signal. BIOT, LaBraM, and SensorLM all fall
behind, likely because BIOT/LaBraM's pretraining never included real ECG
at all, and SensorLM's from-scratch, no-pretraining approach doesn't get
the same benefit From-Scratch/Next-Token get from self-supervised exposure
to the target signal.

**Overall takeaway**: if a wearable can capture EEG, BIOT or LaBraM is the
stronger choice by a clear margin. If it's ECG-only, a SleepFM variant
pretrained on this cohort's real ECG data — From-Scratch or Next-Token —
beats larger, EEG-pretrained models working outside their comfort zone.

## Pending

None -- all 6 models (From-Scratch, Spectral, Next-Token, BIOT, LaBraM,
SensorLM) complete across all 3 modalities (EEG_ONLY, ECG_ONLY, EEG_ECG),
5 folds each.
