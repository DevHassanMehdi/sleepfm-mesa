# MESA 90-Run Benchmark

**Finding:** BIOT leads EEG-based modalities. From-Scratch leads ECG-only by a wide margin.

- All 6 models complete on all 3 modalities, 5 folds each, 90 runs total.
- From-Scratch's ECG-only lead comes mostly from a fixed training schedule, see 03.
- BIOT, LaBraM, and SensorLM never saw MESA ECG during pretraining.

**Setup:** Full MESA cohort, 2056 subjects, fold5_v1. Patience 50 for From-Scratch, Spectral, Next-Token. Patience 20 for BIOT, LaBraM, SensorLM.

**Legend:** 🟩 best in column, bold value.

## EEG Only

| Model | Patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| BIOT | 20 | **0.7541 ± 0.0047** 🟩 | 0.8274 | 0.9455 | **0.5363** 🟩 | 0.8023 | **0.6798** 🟩 | 0.8068 |
| From-Scratch | 50 | 0.7451 ± 0.0052 | **0.8403** 🟩 | **0.9468** 🟩 | 0.4661 | **0.8182** 🟩 | 0.6569 | **0.8373** 🟩 |
| Next-Token | 50 | 0.7394 ± 0.0051 | 0.8373 | 0.9444 | 0.4443 | 0.8158 | 0.6684 | 0.8241 |
| LaBraM | 20 | 0.7301 ± 0.0040 | 0.8108 | 0.9407 | 0.4787 | 0.7866 | 0.6704 | 0.7742 |
| Spectral | 50 | 0.7122 ± 0.0055 | 0.8185 | 0.9339 | 0.3971 | 0.7949 | 0.6298 | 0.8052 |
| SensorLM | 20 | 0.6446 ± 0.0095 | 0.7382 | 0.9085 | 0.3726 | 0.7103 | 0.6365 | 0.5951 |

## ECG Only

| Model | Patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| From-Scratch | 50 | **0.5575 ± 0.0058** 🟩 | **0.7048** 🟩 | **0.8701** 🟩 | **0.2214** 🟩 | **0.6634** 🟩 | **0.4064** 🟩 | **0.6264** 🟩 |
| Next-Token | 50 | 0.4158 ± 0.0074 | 0.5987 | 0.8126 | 0.0369 | 0.5335 | 0.2803 | 0.4156 |
| Spectral | 50 | 0.3402 ± 0.0022 | 0.5106 | 0.7497 | 0.0001 | 0.4327 | 0.2119 | 0.3064 |
| LaBraM | 20 | 0.3390 ± 0.0048 | 0.4822 | 0.7532 | 0.1638 | 0.4215 | 0.1417 | 0.2145 |
| BIOT | 20 | 0.3235 ± 0.0101 | 0.4541 | 0.6999 | 0.1708 | 0.4280 | 0.1378 | 0.1808 |
| SensorLM | 20 | 0.3028 ± 0.0089 | 0.4143 | 0.7012 | 0.1548 | 0.3105 | 0.1378 | 0.2097 |

From-Scratch pretrained on MESA ECG with a contrastive objective. BIOT, LaBraM, and SensorLM did not see MESA ECG during pretraining.

## EEG + ECG

| Model | Patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| BIOT | 20 | **0.7479 ± 0.0044** 🟩 | 0.8289 | **0.9471** 🟩 | **0.5343** 🟩 | 0.8009 | 0.6629 | 0.7944 |
| From-Scratch | 50 | 0.7435 ± 0.0035 | **0.8386** 🟩 | 0.9464 | 0.4720 | **0.8156** 🟩 | 0.6446 | **0.8388** 🟩 |
| Next-Token | 50 | 0.7342 ± 0.0039 | 0.8339 | 0.9433 | 0.4368 | 0.8113 | 0.6530 | 0.8266 |
| LaBraM | 20 | 0.7230 ± 0.0026 | 0.8068 | 0.9404 | 0.4700 | 0.7827 | **0.6640** 🟩 | 0.7578 |
| Spectral | 50 | 0.7149 ± 0.0071 | 0.8202 | 0.9322 | 0.4033 | 0.7983 | 0.6357 | 0.8049 |
| SensorLM | 20 | 0.6425 ± 0.0137 | 0.7352 | 0.9094 | 0.3710 | 0.7086 | 0.6334 | 0.5903 |

### Per-fold macro F1

**EEG Only**

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| BIOT | 0.7551 | 0.7626 | 0.7489 | 0.7508 | 0.7534 |
| From-Scratch | 0.7385 | 0.7496 | 0.7446 | 0.7521 | 0.7404 |
| Next-Token | 0.7350 | 0.7461 | 0.7373 | 0.7449 | 0.7338 |
| LaBraM | 0.7281 | 0.7379 | 0.7294 | 0.7285 | 0.7266 |
| Spectral | 0.7055 | 0.7193 | 0.7097 | 0.7182 | 0.7084 |
| SensorLM | 0.6387 | 0.6597 | 0.6432 | 0.6497 | 0.6318 |

**ECG Only**

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| From-Scratch | 0.5595 | 0.5606 | 0.5601 | 0.5616 | 0.5459 |
| Next-Token | 0.4042 | 0.4198 | 0.4129 | 0.4264 | 0.4156 |
| Spectral | 0.3371 | 0.3413 | 0.3419 | 0.3380 | 0.3425 |
| LaBraM | 0.3312 | 0.3428 | 0.3354 | 0.3424 | 0.3430 |
| BIOT | 0.3038 | 0.3275 | 0.3319 | 0.3257 | 0.3285 |
| SensorLM | 0.3067 | 0.2978 | 0.3024 | 0.2903 | 0.3168 |

**EEG + ECG**

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| BIOT | 0.7507 | 0.7539 | 0.7419 | 0.7491 | 0.7439 |
| From-Scratch | 0.7419 | 0.7493 | 0.7423 | 0.7451 | 0.7388 |
| Next-Token | 0.7344 | 0.7416 | 0.7309 | 0.7331 | 0.7309 |
| LaBraM | 0.7207 | 0.7242 | 0.7224 | 0.7273 | 0.7203 |
| Spectral | 0.7129 | 0.7240 | 0.7122 | 0.7213 | 0.7040 |
| SensorLM | 0.6413 | 0.6637 | 0.6317 | 0.6509 | 0.6253 |

BIOT, LaBraM, and SensorLM per-fold values also exist in their own `*_fullcohort_ALL_results.txt` files and independently in `per_run_metrics/`.
