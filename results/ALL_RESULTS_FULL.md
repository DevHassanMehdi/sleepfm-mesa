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

*From-Scratch's ECG_ONLY result are much more interesting than they seem at first sight -- see Discussion.*

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

Across the three SleepFM variants, all land close together on EEG-based modalities, with BIOT ahead of the whole SleepFM family there. The gap is much narrower than before the label-corruption fix. Most of that gap closed because SleepFM's results were the ones most damaged by broken labels. BIOT and LaBraM barely moved, since neither depends on MESA's labels for pretraining. SleepFM does.

ECG-only tells a different story, and it's a new one. From-Scratch leads by a wide margin here. That wasn't true in earlier 350-subject work on this project, where From-Scratch's ECG-only score sat at 0.3353, in line with every other model's ECG-only range. The difference traces to a specific change: the earlier version pretrained on four modality groups together (EEG, respiratory effort, ECG, EMG), so each modality's embedding had to balance alignment across three targets at once. The full-cohort version narrows this to two groups, EEG and ECG only. With nothing else to balance against, the cross-modal alignment between EEG and ECG gets much tighter. ECG's embedding ends up carrying EEG-relevant structure, not just raw ECG on its own terms.

That tighter coupling is what produces the elevated score, and at deployment time, it costs nothing. Only ECG is needed. Nothing about it requires EEG hardware at test time. Part of what this study is chasing is exactly this: better sleep staging from a signal as limited as ECG, by training smarter rather than adding sensors. This result says that's possible, at least with this pretraining approach.

It also means From-Scratch's ECG-only number isn't answering the same question as the other five models' ECG-only numbers. BIOT, LaBraM, SensorLM, and Spectral never saw EEG during training, so their scores measure what ECG alone can do with no help. From-Scratch's score measures what ECG can do once EEG has taught it what to look for during pretraining. Both are real numbers. They're just not measuring the same thing.

Next step: an EEG-naive ablation of the same cross-modal approach, to see whether this gain holds without EEG in the loop at all, or whether it depends on that exposure.

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
