# Moody / PhysioNet 2026 (I0006)

**Finding:** Moody tracks MESA closely. MESA beats Moody on every condition, for every model, at matched patience.

- The biggest MESA advantage is on ECG-only and BIOT, 0.04 to 0.12 macro F1.
- The smallest MESA advantage is EEG3-only, the one electrode-exact channel, 0.004 to 0.016.
- Retraining on Moody's own 1135 records closes most of the zero-shot gap to MESA.

**Setup:** Site I0006, 1142 recordings downloaded, 1135 used. EEG1 and EEG2 (F4-M1, O2-M1) are stand-ins, not electrode matches. EEG3 (C4-M1) is an exact match to MESA's EEG3. Next-Token's token cache needed `--reseed_empty` to fix one dead cluster. Total GPU-hours, this task: 137.09.

**Legend:** 🟩 best in column, bold value.

Data: BDSP Credentialed Health Data License. Record IDs are not included.

## Zero-shot (MESA-trained, no Moody training)

| Model | Modality | Moody | MESA | Drop |
|---|---|---|---|---|
| From-Scratch | 3-channel EEG | **0.6708 ± 0.0116** 🟩 | 0.7451 ± 0.0052 | -0.0743 |
| Next-Token | 3-channel EEG | 0.6703 ± 0.0085 | 0.7394 ± 0.0051 | -0.0691 |
| Next-Token | EEG3-only | 0.6535 ± 0.0053 | 0.6964 ± 0.0091 | **-0.0429** 🟩 |
| BIOT | 3-channel EEG | 0.6342 ± 0.0094 | **0.7541 ± 0.0047** 🟩 | -0.1199 |
| From-Scratch | EEG3-only | 0.6271 ± 0.0171 | 0.6886 ± 0.0121 | -0.0615 |
| BIOT | EEG3-only | 0.6230 ± 0.0108 | 0.6998 ± 0.0088 | -0.0768 |
| From-Scratch | ECG-only | 0.4691 ± 0.0146 | 0.5575 ± 0.0058 | -0.0884 |
| Next-Token | ECG-only | 0.3372 ± 0.0178 | 0.4158 ± 0.0074 | -0.0786 |
| BIOT | ECG-only | 0.2655 ± 0.0105 | 0.3235 ± 0.0101 | -0.0580 |

### Zero-shot per-class (Moody)

| Model | Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| From-Scratch | 3ch-EEG | 0.6708 ± 0.0116 | 0.7478 ± 0.0133 | 0.8419 | 0.3363 | 0.7765 | 0.6016 | 0.7979 |
| Next-Token | 3ch-EEG | 0.6703 ± 0.0085 | 0.7446 ± 0.0077 | 0.8339 | 0.3487 | 0.7680 | 0.5990 | 0.8021 |
| BIOT | 3ch-EEG | 0.6342 ± 0.0094 | 0.7004 ± 0.0103 | 0.8415 | 0.3164 | 0.7284 | 0.5904 | 0.6945 |
| From-Scratch | EEG3-only | 0.6271 ± 0.0171 | 0.7052 ± 0.0163 | 0.8243 | 0.2630 | 0.7098 | 0.5581 | 0.7801 |
| Next-Token | EEG3-only | 0.6535 ± 0.0053 | 0.7310 ± 0.0096 | 0.8478 | 0.3016 | 0.7395 | 0.5704 | 0.8084 |
| BIOT | EEG3-only | 0.6230 ± 0.0108 | 0.6841 ± 0.0136 | 0.8391 | 0.3156 | 0.7054 | 0.5609 | 0.6941 |
| From-Scratch | ECG-only | 0.4691 ± 0.0146 | 0.5874 ± 0.0077 | 0.6679 | 0.1400 | 0.6309 | 0.3375 | 0.5690 |
| Next-Token | ECG-only | 0.3372 ± 0.0178 | 0.4732 ± 0.0237 | 0.5859 | 0.0340 | 0.5014 | 0.2459 | 0.3191 |
| BIOT | ECG-only | 0.2655 ± 0.0105 | 0.3516 ± 0.0122 | 0.4763 | 0.0920 | 0.4359 | 0.1490 | 0.1744 |

## Retrained, patience 50

| Model | Condition | Moody | MESA (patience 50) | MESA − Moody |
|---|---|---|---|---|
| From-Scratch | EEG_ECG | **0.7145 ± 0.0080** 🟩 | 0.7435 ± 0.0035 | 0.0290 |
| From-Scratch | EEG_ONLY | 0.7143 ± 0.0091 | **0.7451 ± 0.0052** 🟩 | 0.0308 |
| From-Scratch | EEG3_ONLY | 0.7029 ± 0.0104 | 0.7100 ± 0.0021 | 0.0071 |
| Next-Token | EEG_ONLY | 0.7028 ± 0.0084 | 0.7394 ± 0.0051 | 0.0366 |
| Next-Token | EEG_ECG | 0.7015 ± 0.0087 | 0.7342 ± 0.0039 | 0.0327 |
| Next-Token | EEG3_ONLY | 0.6996 ± 0.0066 | n/a, no patience-50 MESA run | n/a |
| From-Scratch | ECG_ONLY | 0.5189 ± 0.0177 | 0.5575 ± 0.0058 | **0.0386** 🟩 |
| Next-Token | ECG_ONLY | 0.3815 ± 0.0065 | 0.4158 ± 0.0074 | 0.0343 |

### Retrained patience-50 per-class (Moody)

| Model | Condition | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| From-Scratch | EEG_ONLY | 0.7143 ± 0.0091 | 0.8057 ± 0.0072 | 0.8858 | 0.3646 | 0.8388 | 0.6313 | 0.8511 |
| From-Scratch | EEG3_ONLY | 0.7029 ± 0.0104 | 0.7971 ± 0.0097 | 0.8787 | 0.3360 | 0.8310 | 0.6265 | 0.8421 |
| From-Scratch | EEG_ECG | 0.7145 ± 0.0080 | 0.8047 ± 0.0080 | 0.8842 | 0.3597 | 0.8370 | 0.6400 | 0.8516 |
| From-Scratch | ECG_ONLY | 0.5189 ± 0.0177 | 0.6494 ± 0.0114 | 0.7410 | 0.0977 | 0.6962 | 0.4447 | 0.6149 |
| Next-Token | EEG_ONLY | 0.7028 ± 0.0084 | -- | 0.8850 | 0.3529 | 0.8272 | 0.6306 | 0.8184 |
| Next-Token | EEG3_ONLY | 0.6996 ± 0.0066 | -- | 0.8796 | 0.3249 | 0.8331 | 0.6375 | 0.8229 |
| Next-Token | EEG_ECG | 0.7015 ± 0.0087 | -- | 0.8815 | 0.3428 | 0.8296 | 0.6311 | 0.8226 |
| Next-Token | ECG_ONLY | 0.3815 ± 0.0065 | -- | 0.6690 | 0.0024 | 0.5749 | 0.3087 | 0.3524 |

## Retrained, patience 20 (matched)

| Model | Condition | Moody | MESA (patience 20) | MESA − Moody | Per-fold diff |
|---|---|---|---|---|---|
| From-Scratch | EEG_ONLY | **0.7131 ± 0.0088** 🟩 | 0.7428 ± 0.0045 | 0.0297 | 0.0251, 0.0367, 0.0233, 0.0500, 0.0133 |
| From-Scratch | EEG_ECG | 0.7104 ± 0.0060 | 0.7395 ± 0.0044 | 0.0291 | 0.0254, 0.0349, 0.0194, 0.0441, 0.0219 |
| Next-Token | EEG_ONLY | 0.7033 ± 0.0080 | 0.7350 ± 0.0051 | 0.0317 | 0.0401, 0.0310, 0.0221, 0.0460, 0.0197 |
| Next-Token | EEG_ECG | 0.7020 ± 0.0076 | 0.7322 ± 0.0024 | 0.0302 | 0.0381, 0.0324, 0.0217, 0.0378, 0.0212 |
| Next-Token | EEG3_ONLY | 0.6983 ± 0.0080 | 0.7138 ± 0.0073 | 0.0155 | 0.0048, 0.0287, 0.0137, 0.0262, 0.0043 |
| From-Scratch | EEG3_ONLY | 0.6955 ± 0.0116 | 0.6998 ± 0.0061 | **0.0043** 🟩 | -0.0094, 0.0219, -0.0018, 0.0212, -0.0102 |
| BIOT | EEG_ONLY | 0.6932 ± 0.0096 | **0.7541 ± 0.0047** 🟩 | 0.0609 | 0.0582, 0.0629, 0.0448, 0.0737, 0.0650 |
| BIOT | EEG_ECG | 0.6832 ± 0.0046 | 0.7479 ± 0.0044 | 0.0647 | 0.0707, 0.0688, 0.0538, 0.0733, 0.0568 |
| BIOT | EEG3_ONLY | 0.6693 ± 0.0057 | 0.6998 ± 0.0088 (p10) | n/a, patience mismatch | -- |
| From-Scratch | ECG_ONLY | 0.5062 ± 0.0132 | 0.5431 ± 0.0126 | 0.0369 | 0.0466, 0.0333, -0.0047, 0.0716, 0.0376 |
| Next-Token | ECG_ONLY | 0.3742 ± 0.0107 | 0.4190 ± 0.0049 | 0.0448 | 0.0566, 0.0536, 0.0255, 0.0545, 0.0338 |
| BIOT | ECG_ONLY | 0.2591 ± 0.0073 | 0.3235 ± 0.0101 | 0.0644 | 0.0570, 0.0641, 0.0663, 0.0606, 0.0738 |

Smallest gap, EEG3_ONLY: +0.004 to +0.016. Largest, ECG_ONLY and BIOT generally: +0.04 to +0.065.

### BIOT best validation macro F1 during training (folds 1-4 mean, Moody)

| Condition | Test (5-fold) | Best val (folds 1-4 mean) |
|---|---|---|
| EEG_ONLY | 0.6932 | 0.6924 |
| EEG3_ONLY | 0.6693 | 0.6700 |
| EEG_ECG | 0.6832 | 0.6831 |
| ECG_ONLY | 0.2591 | 0.2642 |

### BIOT patience-20 per-class (Moody)

| Condition | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG_ONLY | 0.6932 ± 0.0096 | 0.7575 ± 0.0138 | 0.8723 ± 0.0103 | 0.3795 ± 0.0076 | 0.7894 ± 0.0128 | 0.6346 ± 0.0148 | 0.7905 ± 0.0200 |
| EEG3_ONLY | 0.6693 ± 0.0057 | 0.7322 ± 0.0087 | 0.8595 ± 0.0042 | 0.3559 ± 0.0070 | 0.7661 ± 0.0113 | 0.6162 ± 0.0152 | 0.7490 ± 0.0104 |
| EEG_ECG | 0.6832 ± 0.0046 | 0.7471 ± 0.0083 | 0.8634 ± 0.0059 | 0.3637 ± 0.0131 | 0.7854 ± 0.0083 | 0.6206 ± 0.0148 | 0.7829 ± 0.0132 |
| ECG_ONLY | 0.2591 ± 0.0073 | 0.3330 ± 0.0161 | 0.4714 ± 0.0113 | 0.1022 ± 0.0076 | 0.3972 ± 0.0279 | 0.1431 ± 0.0311 | 0.1817 ± 0.0098 |

## EEG minus ECG gap, per model

| Model | MESA | Moody |
|---|---|---|
| BIOT | **0.4306** | **0.4341** 🟩 |
| Next-Token | 0.3160 | 0.3291 |
| From-Scratch | 0.1997 | 0.2069 |

## Flat-record sensitivity (8 of 1135 Moody records, all channels flat at once)

| Setup | Condition | Original mean | Excl.-8 mean | Change |
|---|---|---|---|---|
| From-Scratch p50 | EEG_ONLY | 0.7143 | 0.7170 | 0.0027 |
| From-Scratch p50 | EEG3_ONLY | 0.7029 | 0.7051 | 0.0022 |
| From-Scratch p50 | EEG_ECG | 0.7145 | 0.7170 | 0.0025 |
| From-Scratch p50 | ECG_ONLY | 0.5189 | 0.5206 | 0.0016 |
| From-Scratch p20 | EEG_ONLY | 0.7131 | 0.7157 | 0.0026 |
| From-Scratch p20 | EEG3_ONLY | 0.6955 | 0.6983 | **0.0028** 🟩 |
| From-Scratch p20 | EEG_ECG | 0.7104 | 0.7131 | 0.0027 |
| From-Scratch p20 | ECG_ONLY | 0.5062 | 0.5078 | 0.0016 |
| Next-Token p50 | EEG_ONLY | 0.7028 | 0.7056 | 0.0027 |
| Next-Token p50 | EEG3_ONLY | 0.6996 | 0.7023 | 0.0027 |
| Next-Token p50 | EEG_ECG | 0.7015 | 0.7038 | 0.0022 |
| Next-Token p50 | ECG_ONLY | 0.3815 | 0.3824 | 0.0009 |

Every change is small and positive, removing these records makes the test set slightly easier. BIOT has no raw per-window predictions saved, so it was bounded instead: the 8 records contribute 0.685% of EEG_ONLY test windows, plausibly a similar small move.

## Cognitive impairment, age and sex baseline (1135 records)

| Group | n | Age mean | Age std | Age median | Age range | Female | Male |
|---|---|---|---|---|---|---|---|
| Negative | 1024 | 61.75 | 7.86 | 61.0 | 50-90 | 583 (56.9%) | 441 (43.1%) |
| Positive | 111 | 70.41 | 8.19 | 70.0 | 50-90 | 51 (45.9%) | 60 (54.1%) |

### Logistic regression baseline, repeated stratified 5-fold CV (20 repeats)

| Model | Statistic | Mean | Std | 95% interval |
|---|---|---|---|---|
| Age alone | Plain AUROC | **0.7746** 🟩 | 0.0019 | [0.7701, 0.7766] |
| Age alone | Age-conditioned AUROC (\|Δage\|≤2y) | 0.5155 | 0.0134 | [0.4856, 0.5310] |
| Age + sex | Plain AUROC | 0.7742 | 0.0030 | [0.7689, 0.7785] |
| Age + sex | Age-conditioned AUROC (\|Δage\|≤2y) | **0.5206** 🟩 | 0.0185 | [0.4885, 0.5481] |

The age-conditioned AUROC (~0.52) is the right floor for a signal-based model to beat, not the plain AUROC (~0.77), which is mostly the age confound.

### Per-fold macro F1 (Moody, matched patience 20)

| Model | Condition | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| From-Scratch | EEG_ONLY | 0.7148 | 0.7121 | 0.7167 | 0.6975 | 0.7244 |
| From-Scratch | EEG3_ONLY | 0.7033 | 0.6896 | 0.6982 | 0.6765 | 0.7098 |
| From-Scratch | EEG_ECG | 0.7122 | 0.7111 | 0.7143 | 0.6988 | 0.7156 |
| From-Scratch | ECG_ONLY | 0.5007 | 0.5149 | 0.5241 | 0.4852 | 0.5062 |
| Next-Token | EEG_ONLY | 0.6980 | 0.7089 | 0.7132 | 0.6907 | 0.7055 |
| Next-Token | EEG3_ONLY | 0.6984 | 0.6914 | 0.7096 | 0.6879 | 0.7043 |
| Next-Token | EEG_ECG | 0.6956 | 0.7035 | 0.7094 | 0.6911 | 0.7103 |
| Next-Token | ECG_ONLY | 0.3584 | 0.3703 | 0.3903 | 0.3715 | 0.3806 |
| BIOT | EEG_ONLY | 0.6969 | 0.6997 | 0.7041 | 0.6771 | 0.6884 |
| BIOT | EEG3_ONLY | 0.6675 | 0.6721 | 0.6716 | 0.6593 | 0.6761 |
| BIOT | EEG_ECG | 0.6800 | 0.6851 | 0.6881 | 0.6758 | 0.6871 |
| BIOT | ECG_ONLY | 0.2468 | 0.2634 | 0.2656 | 0.2651 | 0.2547 |
