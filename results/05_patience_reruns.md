# Patience Reruns

**Finding:** Higher early-stopping patience helps a little and does not change which model wins.

- SleepFM From-Scratch's EEG, 1 channel (C4-M1) score rises from 0.69 at patience 10 to 0.71 at patience 50.
- At matched patience 20, MESA beats Moody on every condition for every model, see 06.
- Smallest MESA-Moody gap is EEG, 1 channel (C4-M1), the one electrode-exact channel.

## Names used

| Name in this file | Plain description |
|---|---|
| SleepFM From-Scratch | SleepFM's encoder, trained from random initialization with a contrastive pretraining objective, then fine-tuned for sleep staging |
| SleepFM Next-Token | SleepFM's encoder, pretrained with a next-token-prediction objective, then fine-tuned for sleep staging |
| BIOT | The published BIOT EEG foundation model, fine-tuned for sleep staging |
| EEG, 3 channels | All three EEG leads used together |
| EEG, 1 channel (C4-M1) | A single EEG lead, bipolar C4-M1, the one lead that is an exact electrode match between Moody and MESA |
| EEG + ECG | The three EEG leads plus one ECG lead, together |
| ECG only | The single ECG lead alone, no EEG |
| Early-stopping patience | Epochs without validation improvement allowed before training stops; higher patience trains longer |

**Setup:** Full MESA cohort, 2056 subjects, fold5_v1.

**Legend:** 🟩 best in column, bold value.

## Main results

### Early-stopping patience ladder, EEG, 1 channel (C4-M1)

| Model | Patience 10 | Patience 20 | Patience 50 |
|---|---|---|---|
| BIOT | **0.6998 ± 0.0088** 🟩 | -- | -- |
| SleepFM Next-Token | 0.6964 ± 0.0091 | 0.7138 ± 0.0073 | -- |
| SleepFM From-Scratch | 0.6886 ± 0.0121 | 0.6998 ± 0.0061 | **0.7100 ± 0.0021** 🟩 |

"--" means that early-stopping patience was not run for that model.

### Matched patience-20, SleepFM From-Scratch

| Condition | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG, 3 channels | **0.7428 ± 0.0045** 🟩 | **0.8392** 🟩 | **0.9467** 🟩 | **0.4609** 🟩 | **0.8165** 🟩 | 0.6551 | **0.8348** 🟩 |
| EEG + ECG | 0.7395 ± 0.0044 | 0.8377 | 0.9459 | 0.4470 | 0.8144 | **0.6607** 🟩 | 0.8297 |
| EEG, 1 channel (C4-M1) | 0.6998 ± 0.0061 | 0.8060 | 0.9202 | 0.3699 | 0.7830 | 0.6428 | 0.7833 |
| ECG only | 0.5431 ± 0.0126 | 0.7027 | 0.8717 | 0.1415 | 0.6505 | 0.4231 | 0.6288 |

### Matched patience-20, SleepFM Next-Token

| Condition | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG, 3 channels | **0.7350 ± 0.0051** 🟩 | **0.8359** 🟩 | **0.9448** 🟩 | 0.4300 | **0.8149** 🟩 | **0.6684** 🟩 | 0.8171 |
| EEG + ECG | 0.7322 ± 0.0024 | 0.8322 | 0.9439 | 0.4295 | 0.8078 | 0.6631 | **0.8169** 🟩 |
| EEG, 1 channel (C4-M1) | 0.7138 ± 0.0073 | 0.8159 | 0.9256 | 0.3953 | 0.7925 | 0.6589 | 0.7969 |
| ECG only | 0.4190 ± 0.0049 | 0.5980 | 0.8264 | **0.0106** | 0.5147 | 0.3120 | 0.4314 |

### MESA vs Moody, early-stopping patience 20

| Model | Condition | MESA | Moody | MESA − Moody |
|---|---|---|---|---|
| SleepFM From-Scratch | EEG, 3 channels | **0.7428 ± 0.0045** | 0.7131 ± 0.0088 | +0.0297 |
| SleepFM From-Scratch | EEG + ECG | 0.7395 ± 0.0044 | 0.7104 ± 0.0060 | +0.0291 |
| BIOT | EEG, 3 channels | 0.7541 ± 0.0047 | 0.6932 ± 0.0096 | +0.0609 |
| BIOT | EEG + ECG | 0.7479 ± 0.0044 | 0.6832 ± 0.0046 | **+0.0647** 🟩 |
| SleepFM Next-Token | EEG, 3 channels | 0.7350 ± 0.0051 | 0.7033 ± 0.0080 | +0.0317 |
| SleepFM Next-Token | EEG + ECG | 0.7322 ± 0.0024 | 0.7020 ± 0.0076 | +0.0302 |
| BIOT | ECG only | 0.3235 ± 0.0101 | 0.2591 ± 0.0073 | 0.0644 |
| SleepFM From-Scratch | ECG only | 0.5431 ± 0.0126 | 0.5062 ± 0.0132 | 0.0369 |
| SleepFM Next-Token | ECG only | 0.4190 ± 0.0049 | 0.3742 ± 0.0107 | 0.0448 |
| SleepFM From-Scratch | EEG, 1 channel (C4-M1) | 0.6998 ± 0.0061 | 0.6955 ± 0.0116 | 0.0043 |
| SleepFM Next-Token | EEG, 1 channel (C4-M1) | 0.7138 ± 0.0073 | 0.6983 ± 0.0080 | **0.0155** |
| BIOT | EEG, 1 channel (C4-M1) | 0.6998 ± 0.0088 (patience 10) | 0.6693 ± 0.0057 (patience 20) | n/a, patience mismatch |

EEG, 1 channel (C4-M1) has the smallest MESA − Moody gap of every condition. ECG only and BIOT generally have the largest.

## Detail tables

### Per-fold macro F1

**EEG, 1 channel (C4-M1), early-stopping patience ladder**

| Model | Early-stopping patience | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| SleepFM From-Scratch | 10 | 0.6966 | 0.6947 | 0.6947 | 0.6923 | 0.6646 |
| SleepFM From-Scratch | 20 | 0.6939 | 0.7115 | 0.6964 | 0.6977 | 0.6996 |
| SleepFM From-Scratch | 50 | 0.7103 | 0.7137 | 0.7087 | 0.7102 | 0.7072 |
| SleepFM Next-Token | 10 | 0.6981 | 0.7051 | 0.6905 | 0.7060 | 0.6821 |
| SleepFM Next-Token | 20 | 0.7032 | 0.7201 | 0.7233 | 0.7141 | 0.7086 |
| BIOT | 10 | 0.6937 | 0.7164 | 0.6980 | 0.6996 | 0.6915 |

**Matched patience-20, other conditions**

| Model | Condition | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG, 3 channels | 0.7399 | 0.7488 | 0.7400 | 0.7475 | 0.7377 |
| SleepFM From-Scratch | EEG + ECG | 0.7376 | 0.7460 | 0.7337 | 0.7429 | 0.7375 |
| SleepFM From-Scratch | ECG only | 0.5473 | 0.5482 | 0.5194 | 0.5568 | 0.5438 |
| SleepFM Next-Token | EEG, 3 channels | 0.7381 | 0.7399 | 0.7353 | 0.7367 | 0.7252 |
| SleepFM Next-Token | EEG + ECG | 0.7337 | 0.7359 | 0.7311 | 0.7289 | 0.7315 |
| SleepFM Next-Token | ECG only | 0.4150 | 0.4239 | 0.4158 | 0.4260 | 0.4144 |
