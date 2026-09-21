# SleepFM on MESA: Full-Cohort Results (2056 subjects, fold5_v1)

## Configuration

| Setting | Value |
|---|---|
| Cohort size | 2056 subjects |
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
| BIOT | **0.7541 ± 0.0047** | 0.8274 | 0.9455 | **0.5363** | 0.8023 | **0.6798** | 0.8068 |
| From-Scratch | 0.7451 ± 0.0052 | **0.8403** | **0.9468** | 0.4661 | **0.8182** | 0.6798 | **0.8373** |
| Next-Token | 0.7394 ± 0.0051 | 0.8373 | 0.9444 | 0.4443 | 0.8158 | 0.6684 | 0.8241 |
| LaBraM | 0.7301 ± 0.0040 | 0.8108 | 0.9407 | 0.4787 | 0.7866 | 0.6704 | 0.7742 |
| Spectral | 0.7122 ± 0.0055 | 0.8185 | 0.9339 | 0.3971 | 0.7949 | 0.6298 | 0.8052 |
| SensorLM | 0.6446 ± 0.0095 | 0.7382 | 0.9085 | 0.3726 | 0.7103 | 0.6365 | 0.5951 |

## ECG Only

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| From-Scratch | **0.5575 ± 0.0058** | **0.7048** | **0.8701** | **0.2214** | **0.6634** | **0.4064** | **0.6264** |
| Next-Token | 0.4158 ± 0.0074 | 0.5987 | 0.8126 | 0.0369 | 0.5335 | 0.2803 | 0.4156 |
| Spectral | 0.3402 ± 0.0022 | 0.5106 | 0.7497 | 0.0001 | 0.4327 | 0.2119 | 0.3064 |
| LaBraM | 0.3390 ± 0.0048 | 0.4822 | 0.7532 | 0.1638 | 0.4215 | 0.1417 | 0.2145 |
| BIOT | 0.3235 ± 0.0101 | 0.4541 | 0.6999 | 0.1708 | 0.4280 | 0.1378 | 0.1808 |
| SensorLM | 0.3028 ± 0.0089 | 0.4143 | 0.7012 | 0.1548 | 0.3105 | 0.1378 | 0.2097 |

*From-Scratch's ECG_ONLY result is not on the same methodological footing as the other 5 models -- see Discussion.*

## EEG + ECG

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| BIOT | **0.7479 ± 0.0044** | 0.8289 | **0.9471** | 0.5343 | 0.8009 | 0.6629 | 0.7944 |
| From-Scratch | 0.7435 ± 0.0035 | **0.8386** | 0.9464 | **0.4720** | **0.8156** | 0.6446 | **0.8388** |
| Next-Token | 0.7342 ± 0.0039 | 0.8339 | 0.9433 | 0.4368 | 0.8113 | 0.6530 | 0.8266 |
| LaBraM | 0.7230 ± 0.0026 | 0.8068 | 0.9404 | 0.4700 | 0.7827 | **0.6640** | 0.7578 |
| Spectral | 0.7149 ± 0.0071 | 0.8202 | 0.9322 | 0.4033 | 0.7983 | 0.6357 | 0.8049 |
| SensorLM | 0.6425 ± 0.0137 | 0.7352 | 0.9094 | 0.3710 | 0.7086 | 0.6334 | 0.5903 |

## Discussion

**BIOT and From-Scratch now lead the EEG-based modalities (EEG_ONLY,
EEG_ECG), and the gap between externally-pretrained and from-scratch
models has narrowed dramatically** compared to earlier (pre-label-fix)
results. BIOT holds the top macro F1 on both EEG_ONLY (0.7541) and
EEG_ECG (0.7479), but From-Scratch is now close behind on both (0.7451,
0.7435) and actually leads on accuracy and three of five per-class
scores in EEG_ECG. Next-Token and Spectral cluster just below, both
competitive with LaBraM. SensorLM is consistently the weakest performer
across all three modalities (macro F1 in the 0.64 range vs 0.71-0.75 for
the other five) -- its one-stage, no-separate-pretraining design appears
to be a real disadvantage on this task, not a training artifact (all
SensorLM folds converged normally, patience firing as expected).

**ECG_ONLY: From-Scratch leads every single column by a wide margin** --
macro F1 0.5575 vs the next-best (Next-Token) at 0.4158, roughly 14
points ahead, and it wins Wake/N1/N2/N3/REM individually, not just the
aggregate. Before trusting this number, we investigated it directly
(embeddings spot-check, cross-modal correlation, per-class breakdown,
config/job-log audit) to rule out a channel-selection or embeddings
mix-up bug -- see the caveat below. Excluding From-Scratch, Next-Token
holds second place, followed by LaBraM, Spectral, BIOT, and SensorLM
clustered closely together (macro F1 0.30-0.34) -- none of the
externally-EEG-pretrained or no-separate-pretraining models get much
traction on ECG-only signal.

**Caveat: From-Scratch's ECG_ONLY result is not directly comparable to
the other 5 models on the same methodological footing.** From-Scratch's
`leave_one_out` pretraining objective is a cross-modal contrastive
scheme: it explicitly trains each modality's embedding to be predictive
of the other modality's embedding for the same time window. We measured
this directly -- the cross-modal correlation between From-Scratch's
EEG_ONLY and EKG embeddings for the same subjects is 0.50-0.61, versus
0.29 for Spectral's independently-trained per-modality embeddings on the
same subjects. This means From-Scratch's "ECG_ONLY" embeddings carry
real EEG-correlated information injected during pretraining, not signal
isolated to the ECG channel alone. The elevated ECG_ONLY score therefore
reflects the benefit of cross-modally-informed embeddings, not what a
truly ECG-only-trained model can achieve. This is flagged for discussion
and interpretation, not treated as a data or pipeline bug -- the
per-class breakdown showed broad, proportional gains across all five
sleep stages (not concentrated in one suspicious class), and the
absolute score still sits well below any model's EEG_ONLY ceiling,
consistent with genuinely-informed-but-still-ECG-derived embeddings
rather than an outright data leak.

**Overall takeaway**: if a wearable can capture EEG, BIOT and
From-Scratch are now essentially tied for the strongest choice, with
Next-Token and Spectral close behind and LaBraM competitive as well --
the six models are far closer together than before the label-corruption
fix. If it's ECG-only, Next-Token is the most directly comparable
best performer among models trained under standard (non-cross-modal)
objectives; From-Scratch's ECG_ONLY number is real but should be read
with the cross-modal-embedding caveat above in mind.

## Pending

None -- all 6 models (From-Scratch, Spectral, Next-Token, BIOT, LaBraM,
SensorLM) complete across all 3 modalities (EEG_ONLY, ECG_ONLY, EEG_ECG),
5 folds each. 90/90 fine-tuning runs independently verified: raw
prediction pickles reconstructed directly for From-Scratch/Spectral/
Next-Token, per_subject_results.csv + classification_report.txt
reconstruction for BIOT/LaBraM/SensorLM (no raw pickles persisted by
that pipeline). Zero mismatches. Fold-disjointness confirmed for all 18
model/modality combinations: exactly 2056 unique subjects, zero overlap
across each combination's 5 folds.
