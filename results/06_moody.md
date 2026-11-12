# Moody / PhysioNet 2026 (I0006)

**Finding:** Moody tracks MESA closely. MESA beats Moody on every condition, for every model, at matched patience.

- The biggest MESA advantage is on ECG only and BIOT, 0.04 to 0.12.
- The smallest MESA advantage is EEG, 1 channel (C4-M1), the one electrode-exact channel, 0.004 to 0.016.
- Retraining on Moody's own 1135 records closes most of the zero-shot gap to MESA.

**Setup:** Site I0006, 1142 recordings downloaded, 1135 used. EEG1 and EEG2 (F4-M1, O2-M1) are stand-ins, not electrode matches. EEG3 (C4-M1) is an exact match to MESA's EEG3, and is the channel meant everywhere this file says "EEG, 1 channel (C4-M1)". SleepFM Next-Token's token cache needed `--reseed_empty` to fix one dead cluster. Total GPU-hours, this task: 137.09.

**Legend:** 🟩 best in column, bold value.

Data: BDSP Credentialed Health Data License. Record IDs are not included.

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
| MESA | Trained and tested on the MESA cohort (2056 subjects) |
| Moody zero-shot | A MESA-trained model evaluated directly on Moody, with no additional training on Moody data |
| Moody retrained | The same architecture, trained from scratch on Moody's own 1135 subjects and tested on Moody |
| Early-stopping patience | Epochs without validation improvement allowed before training stops; higher patience trains longer |

## Main results

| Condition | Model | MESA (trained and tested on MESA) | Moody zero-shot (trained on MESA, tested on Moody) | Moody retrained (trained and tested on Moody) | Drop: MESA to zero-shot | Drop: MESA to retrained |
|---|---|---|---|---|---|---|
| EEG, 3 channels | SleepFM From-Scratch | 0.745 | **0.671** 🟩 | **0.714** 🟩 | 0.074 | 0.031 |
| EEG, 3 channels | SleepFM Next-Token | 0.739 | 0.670 | 0.703 | 0.069 | 0.036 |
| EEG, 3 channels | BIOT | **0.754** 🟩 | 0.634 | 0.693 | 0.120 | 0.061 |
| EEG, 1 channel (C4-M1) | SleepFM From-Scratch | 0.710 | 0.627 | **0.703** 🟩 | 0.083 | 0.007 |
| EEG, 1 channel (C4-M1) | SleepFM Next-Token | **0.714** 🟩 | **0.653** 🟩 | 0.700 | 0.060 | 0.014 |
| EEG, 1 channel (C4-M1) | BIOT | 0.700 | 0.623 | 0.669 | 0.077 | 0.030 \*\* |
| EEG + ECG | SleepFM From-Scratch | 0.744 | 0.665 | **0.715** 🟩 | 0.079 | 0.029 |
| EEG + ECG | SleepFM Next-Token | 0.734 | **0.671** 🟩 | 0.702 | 0.063 | 0.032 |
| EEG + ECG | BIOT | **0.748** 🟩 | 0.623 | 0.683 | 0.125 | 0.065 |
| ECG only | SleepFM From-Scratch | **0.557** 🟩 | **0.469** 🟩 | **0.519** 🟩 | 0.088 | 0.039 |
| ECG only | SleepFM Next-Token | 0.419 | 0.337 | 0.382 | 0.082 | 0.037 |
| ECG only | BIOT | 0.324 | 0.266 | 0.259 | 0.058 | 0.064 |

Macro F1, mean of 5 folds, 3 decimals. "MESA" in this table is the best macro F1 among the MESA runs that exist for that cell: patience 20 or 50 for SleepFM From-Scratch/Next-Token, and patience 10 for BIOT's EEG-1-channel MESA run (BIOT has no patience-20 run for that condition). "Moody retrained" is likewise the best among the Moody retrain runs that exist (patience 20 or 50; BIOT only has patience 20). The detail tables below give every run and which one was picked for each cell of this table.

\* The MESA-trained models behind the two EEG-1-channel zero-shot rows (SleepFM From-Scratch, SleepFM Next-Token) were trained at patience 10, confirmed from each model's saved `config.json`, not at the patience shown in this table's MESA column (patience 50 for SleepFM From-Scratch, patience 20 for SleepFM Next-Token -- the best MESA run that exists for that condition, used as this table's MESA baseline). Comparing each zero-shot model only against its own, same-checkpoint MESA test-fold score instead gives: SleepFM From-Scratch 0.689 (MESA, patience 10) vs 0.627 (Moody zero-shot), drop 0.061; SleepFM Next-Token 0.696 (MESA, patience 10) vs 0.653 (Moody zero-shot), drop 0.043. Both same-model drops are about 0.02 smaller than the 0.083 and 0.060 shown in the table, because the table's MESA baseline is a higher-patience, better-scoring run than the one actually used zero-shot.

\*\* BIOT's EEG-1-channel retrained drop compares a MESA run trained at patience 10 (its only MESA run for this condition) against a Moody run trained at patience 20. The two patiences do not match; the detail tables below mark this row the same way.

## Zero-shot detail (MESA-trained, no Moody training)

| Model | Condition | Early-stopping patience (epochs without improvement before stopping) | Moody zero-shot | MESA (same model, own test fold) | Drop (Moody − MESA) |
|---|---|---|---|---|---|
| SleepFM Next-Token | EEG + ECG | 50 | **0.6714 ± 0.0059** 🟩 | 0.7342 ± 0.0039 | -0.0628 |
| SleepFM From-Scratch | EEG, 3 channels | 50 | 0.6708 ± 0.0116 | 0.7451 ± 0.0052 | -0.0743 |
| SleepFM Next-Token | EEG, 3 channels | 50 | 0.6703 ± 0.0085 | 0.7394 ± 0.0051 | -0.0691 |
| SleepFM From-Scratch | EEG + ECG | 50 | 0.6648 ± 0.0049 | 0.7435 ± 0.0035 | -0.0787 |
| SleepFM Next-Token | EEG, 1 channel (C4-M1) | 10 | 0.6535 ± 0.0053 | 0.6964 ± 0.0091 | **-0.0429** 🟩 |
| BIOT | EEG, 3 channels | 20 | 0.6342 ± 0.0094 | **0.7541 ± 0.0047** 🟩 | -0.1199 |
| SleepFM From-Scratch | EEG, 1 channel (C4-M1) | 10 | 0.6271 ± 0.0171 | 0.6886 ± 0.0121 | -0.0615 |
| BIOT | EEG + ECG | 20 | 0.6233 ± 0.0047 | 0.7479 ± 0.0044 | -0.1246 |
| BIOT | EEG, 1 channel (C4-M1) | 10 | 0.6230 ± 0.0108 | 0.6998 ± 0.0088 | -0.0768 |
| SleepFM From-Scratch | ECG only | 50 | 0.4691 ± 0.0146 | 0.5575 ± 0.0058 | -0.0884 |
| SleepFM Next-Token | ECG only | 50 | 0.3372 ± 0.0178 | 0.4158 ± 0.0074 | -0.0786 |
| BIOT | ECG only | 20 | 0.2655 ± 0.0105 | 0.3235 ± 0.0101 | -0.0580 |

The EEG + ECG row is new: it was never run before, so these three rows come from a fresh evaluation-only run (job 3029771, no training; the MESA-trained classifier heads for all 5 folds of all 3 models evaluated once each on all 1135 Moody records). Its raw results are in `zeroshot_eegecg_results.json` (SleepFM From-Scratch, SleepFM Next-Token) and `zeroshot_biot_eegecg_results.json` (BIOT). The other nine rows are unchanged from the earlier zero-shot run, archived at `moody2026_pilot/.zeroshot_results.json`, which holds only the aggregate (no per-fold values) for those nine.

### Zero-shot per-class detail (Moody)

| Model | Condition | Early-stopping patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG, 3 channels | 50 | 0.6708 ± 0.0116 | 0.7478 ± 0.0133 | 0.8419 | 0.3363 | 0.7765 | 0.6016 | 0.7979 |
| SleepFM Next-Token | EEG, 3 channels | 50 | 0.6703 ± 0.0085 | 0.7446 ± 0.0077 | 0.8339 | 0.3487 | 0.7680 | 0.5990 | 0.8021 |
| BIOT | EEG, 3 channels | 20 | 0.6342 ± 0.0094 | 0.7004 ± 0.0103 | 0.8415 | 0.3164 | 0.7284 | 0.5904 | 0.6945 |
| SleepFM From-Scratch | EEG, 1 channel (C4-M1) | 10 | 0.6271 ± 0.0171 | 0.7052 ± 0.0163 | 0.8243 | 0.2630 | 0.7098 | 0.5581 | 0.7801 |
| SleepFM Next-Token | EEG, 1 channel (C4-M1) | 10 | 0.6535 ± 0.0053 | 0.7310 ± 0.0096 | 0.8478 | 0.3016 | 0.7395 | 0.5704 | 0.8084 |
| BIOT | EEG, 1 channel (C4-M1) | 10 | 0.6230 ± 0.0108 | 0.6841 ± 0.0136 | 0.8391 | 0.3156 | 0.7054 | 0.5609 | 0.6941 |
| SleepFM From-Scratch | EEG + ECG | 50 | 0.6648 ± 0.0049 | 0.7426 ± 0.0047 | 0.8391 | 0.3302 | 0.7706 | 0.5964 | 0.7875 |
| SleepFM Next-Token | EEG + ECG | 50 | 0.6714 ± 0.0059 | 0.7514 ± 0.0111 | 0.8429 | 0.3341 | 0.7781 | 0.6078 | 0.7941 |
| BIOT | EEG + ECG | 20 | 0.6233 ± 0.0047 | 0.7063 ± 0.0047 | 0.8335 | 0.2805 | 0.7367 | 0.5880 | 0.6780 |
| SleepFM From-Scratch | ECG only | 50 | 0.4691 ± 0.0146 | 0.5874 ± 0.0077 | 0.6679 | 0.1400 | 0.6309 | 0.3375 | 0.5690 |
| SleepFM Next-Token | ECG only | 50 | 0.3372 ± 0.0178 | 0.4732 ± 0.0237 | 0.5859 | 0.0340 | 0.5014 | 0.2459 | 0.3191 |
| BIOT | ECG only | 20 | 0.2655 ± 0.0105 | 0.3516 ± 0.0122 | 0.4763 | 0.0920 | 0.4359 | 0.1490 | 0.1744 |

The EEG + ECG rows are new (same job 3029771 as above); the other nine are unchanged.

### Zero-shot per-fold macro F1, EEG + ECG only

The other eight zero-shot conditions above have no saved per-fold detail (the earlier run kept only the aggregate). EEG + ECG is new and kept per-fold values:

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| SleepFM From-Scratch | 0.6621 | 0.6648 | 0.6740 | 0.6597 | 0.6632 |
| SleepFM Next-Token | 0.6640 | 0.6787 | 0.6781 | 0.6683 | 0.6678 |
| BIOT | 0.6217 | 0.6273 | 0.6160 | 0.6294 | 0.6223 |

## Retrained, patience 50

| Model | Condition | Early-stopping patience | Moody | MESA | MESA − Moody |
|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG + ECG | 50 | **0.7145 ± 0.0080** 🟩 | 0.7435 ± 0.0035 | 0.0290 |
| SleepFM From-Scratch | EEG, 3 channels | 50 | 0.7143 ± 0.0091 | **0.7451 ± 0.0052** 🟩 | 0.0308 |
| SleepFM From-Scratch | EEG, 1 channel (C4-M1) | 50 | 0.7029 ± 0.0104 | 0.7100 ± 0.0021 | 0.0071 |
| SleepFM Next-Token | EEG, 3 channels | 50 | 0.7028 ± 0.0084 | 0.7394 ± 0.0051 | 0.0366 |
| SleepFM Next-Token | EEG + ECG | 50 | 0.7015 ± 0.0087 | 0.7342 ± 0.0039 | 0.0327 |
| SleepFM Next-Token | EEG, 1 channel (C4-M1) | 50 | 0.6996 ± 0.0066 | n/a, no patience-50 MESA run | n/a |
| SleepFM From-Scratch | ECG only | 50 | 0.5189 ± 0.0177 | 0.5575 ± 0.0058 | **0.0386** 🟩 |
| SleepFM Next-Token | ECG only | 50 | 0.3815 ± 0.0065 | 0.4158 ± 0.0074 | 0.0343 |

### Retrained patience-50 per-class detail (Moody)

| Model | Condition | Early-stopping patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG, 3 channels | 50 | 0.7143 ± 0.0091 | 0.8057 ± 0.0072 | 0.8858 | 0.3646 | 0.8388 | 0.6313 | 0.8511 |
| SleepFM From-Scratch | EEG, 1 channel (C4-M1) | 50 | 0.7029 ± 0.0104 | 0.7971 ± 0.0097 | 0.8787 | 0.3360 | 0.8310 | 0.6265 | 0.8421 |
| SleepFM From-Scratch | EEG + ECG | 50 | 0.7145 ± 0.0080 | 0.8047 ± 0.0080 | 0.8842 | 0.3597 | 0.8370 | 0.6400 | 0.8516 |
| SleepFM From-Scratch | ECG only | 50 | 0.5189 ± 0.0177 | 0.6494 ± 0.0114 | 0.7410 | 0.0977 | 0.6962 | 0.4447 | 0.6149 |
| SleepFM Next-Token | EEG, 3 channels | 50 | 0.7028 ± 0.0084 | -- | 0.8850 | 0.3529 | 0.8272 | 0.6306 | 0.8184 |
| SleepFM Next-Token | EEG, 1 channel (C4-M1) | 50 | 0.6996 ± 0.0066 | -- | 0.8796 | 0.3249 | 0.8331 | 0.6375 | 0.8229 |
| SleepFM Next-Token | EEG + ECG | 50 | 0.7015 ± 0.0087 | -- | 0.8815 | 0.3428 | 0.8296 | 0.6311 | 0.8226 |
| SleepFM Next-Token | ECG only | 50 | 0.3815 ± 0.0065 | -- | 0.6690 | 0.0024 | 0.5749 | 0.3087 | 0.3524 |

## Retrained, patience 20 (matched)

| Model | Condition | Early-stopping patience | Moody | MESA | MESA − Moody | Per-fold diff |
|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG, 3 channels | 20 | **0.7131 ± 0.0088** 🟩 | 0.7428 ± 0.0045 | 0.0297 | 0.0251, 0.0367, 0.0233, 0.0500, 0.0133 |
| SleepFM From-Scratch | EEG + ECG | 20 | 0.7104 ± 0.0060 | 0.7395 ± 0.0044 | 0.0291 | 0.0254, 0.0349, 0.0194, 0.0441, 0.0219 |
| SleepFM Next-Token | EEG, 3 channels | 20 | 0.7033 ± 0.0080 | 0.7350 ± 0.0051 | 0.0317 | 0.0401, 0.0310, 0.0221, 0.0460, 0.0197 |
| SleepFM Next-Token | EEG + ECG | 20 | 0.7020 ± 0.0076 | 0.7322 ± 0.0024 | 0.0302 | 0.0381, 0.0324, 0.0217, 0.0378, 0.0212 |
| SleepFM Next-Token | EEG, 1 channel (C4-M1) | 20 | 0.6983 ± 0.0080 | 0.7138 ± 0.0073 | 0.0155 | 0.0048, 0.0287, 0.0137, 0.0262, 0.0043 |
| SleepFM From-Scratch | EEG, 1 channel (C4-M1) | 20 | 0.6955 ± 0.0116 | 0.6998 ± 0.0061 | **0.0043** 🟩 | -0.0094, 0.0219, -0.0018, 0.0212, -0.0102 |
| BIOT | EEG, 3 channels | 20 | 0.6932 ± 0.0096 | **0.7541 ± 0.0047** 🟩 | 0.0609 | 0.0582, 0.0629, 0.0448, 0.0737, 0.0650 |
| BIOT | EEG + ECG | 20 | 0.6832 ± 0.0046 | 0.7479 ± 0.0044 | 0.0647 | 0.0707, 0.0688, 0.0538, 0.0733, 0.0568 |
| BIOT | EEG, 1 channel (C4-M1) | 20 (MESA: 10) \*\* | 0.6693 ± 0.0057 | 0.6998 ± 0.0088 | n/a, patience mismatch | -- |
| SleepFM From-Scratch | ECG only | 20 | 0.5062 ± 0.0132 | 0.5431 ± 0.0126 | 0.0369 | 0.0466, 0.0333, -0.0047, 0.0716, 0.0376 |
| SleepFM Next-Token | ECG only | 20 | 0.3742 ± 0.0107 | 0.4190 ± 0.0049 | 0.0448 | 0.0566, 0.0536, 0.0255, 0.0545, 0.0338 |
| BIOT | ECG only | 20 | 0.2591 ± 0.0073 | 0.3235 ± 0.0101 | 0.0644 | 0.0570, 0.0641, 0.0663, 0.0606, 0.0738 |

Smallest gap, EEG, 1 channel (C4-M1): +0.004 to +0.016. Largest, ECG only and BIOT generally: +0.04 to +0.065. \*\* See the main-results footnote above: BIOT's only MESA run for this condition used patience 10.

### BIOT best validation macro F1 during training (folds 1-4 mean, Moody)

| Condition | Test (5-fold) | Best val (folds 1-4 mean) |
|---|---|---|
| EEG, 3 channels | 0.6932 | 0.6924 |
| EEG, 1 channel (C4-M1) | 0.6693 | 0.6700 |
| EEG + ECG | 0.6832 | 0.6831 |
| ECG only | 0.2591 | 0.2642 |

### BIOT per-class detail, retrained (Moody)

| Condition | Early-stopping patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| EEG, 3 channels | 20 | 0.6932 ± 0.0096 | 0.7575 ± 0.0138 | 0.8723 ± 0.0103 | 0.3795 ± 0.0076 | 0.7894 ± 0.0128 | 0.6346 ± 0.0148 | 0.7905 ± 0.0200 |
| EEG, 1 channel (C4-M1) | 20 | 0.6693 ± 0.0057 | 0.7322 ± 0.0087 | 0.8595 ± 0.0042 | 0.3559 ± 0.0070 | 0.7661 ± 0.0113 | 0.6162 ± 0.0152 | 0.7490 ± 0.0104 |
| EEG + ECG | 20 | 0.6832 ± 0.0046 | 0.7471 ± 0.0083 | 0.8634 ± 0.0059 | 0.3637 ± 0.0131 | 0.7854 ± 0.0083 | 0.6206 ± 0.0148 | 0.7829 ± 0.0132 |
| ECG only | 20 | 0.2591 ± 0.0073 | 0.3330 ± 0.0161 | 0.4714 ± 0.0113 | 0.1022 ± 0.0076 | 0.3972 ± 0.0279 | 0.1431 ± 0.0311 | 0.1817 ± 0.0098 |

## EEG minus ECG gap, per model

| Model | MESA | Moody |
|---|---|---|
| BIOT | **0.4306** | **0.4341** 🟩 |
| SleepFM Next-Token | 0.3160 | 0.3291 |
| SleepFM From-Scratch | 0.1997 | 0.2069 |

## Flat-record sensitivity (8 of 1135 Moody records, all channels flat at once)

| Model | Early-stopping patience | Condition | Original mean | Excl.-8 mean | Change |
|---|---|---|---|---|---|
| SleepFM From-Scratch | 50 | EEG, 3 channels | 0.7143 | 0.7170 | 0.0027 |
| SleepFM From-Scratch | 50 | EEG, 1 channel (C4-M1) | 0.7029 | 0.7051 | 0.0022 |
| SleepFM From-Scratch | 50 | EEG + ECG | 0.7145 | 0.7170 | 0.0025 |
| SleepFM From-Scratch | 50 | ECG only | 0.5189 | 0.5206 | 0.0016 |
| SleepFM From-Scratch | 20 | EEG, 3 channels | 0.7131 | 0.7157 | 0.0026 |
| SleepFM From-Scratch | 20 | EEG, 1 channel (C4-M1) | 0.6955 | 0.6983 | **0.0028** 🟩 |
| SleepFM From-Scratch | 20 | EEG + ECG | 0.7104 | 0.7131 | 0.0027 |
| SleepFM From-Scratch | 20 | ECG only | 0.5062 | 0.5078 | 0.0016 |
| SleepFM Next-Token | 50 | EEG, 3 channels | 0.7028 | 0.7056 | 0.0027 |
| SleepFM Next-Token | 50 | EEG, 1 channel (C4-M1) | 0.6996 | 0.7023 | 0.0027 |
| SleepFM Next-Token | 50 | EEG + ECG | 0.7015 | 0.7038 | 0.0022 |
| SleepFM Next-Token | 50 | ECG only | 0.3815 | 0.3824 | 0.0009 |

Every change is small and positive, removing these records makes the test set slightly easier. BIOT has no raw per-window predictions saved, so it was bounded instead: the 8 records contribute 0.685% of EEG, 3 channels test windows, plausibly a similar small move.

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

| Model | Condition | Early-stopping patience | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|---|
| SleepFM From-Scratch | EEG, 3 channels | 20 | 0.7148 | 0.7121 | 0.7167 | 0.6975 | 0.7244 |
| SleepFM From-Scratch | EEG, 1 channel (C4-M1) | 20 | 0.7033 | 0.6896 | 0.6982 | 0.6765 | 0.7098 |
| SleepFM From-Scratch | EEG + ECG | 20 | 0.7122 | 0.7111 | 0.7143 | 0.6988 | 0.7156 |
| SleepFM From-Scratch | ECG only | 20 | 0.5007 | 0.5149 | 0.5241 | 0.4852 | 0.5062 |
| SleepFM Next-Token | EEG, 3 channels | 20 | 0.6980 | 0.7089 | 0.7132 | 0.6907 | 0.7055 |
| SleepFM Next-Token | EEG, 1 channel (C4-M1) | 20 | 0.6984 | 0.6914 | 0.7096 | 0.6879 | 0.7043 |
| SleepFM Next-Token | EEG + ECG | 20 | 0.6956 | 0.7035 | 0.7094 | 0.6911 | 0.7103 |
| SleepFM Next-Token | ECG only | 20 | 0.3584 | 0.3703 | 0.3903 | 0.3715 | 0.3806 |
| BIOT | EEG, 3 channels | 20 | 0.6969 | 0.6997 | 0.7041 | 0.6771 | 0.6884 |
| BIOT | EEG, 1 channel (C4-M1) | 20 | 0.6675 | 0.6721 | 0.6716 | 0.6593 | 0.6761 |
| BIOT | EEG + ECG | 20 | 0.6800 | 0.6851 | 0.6881 | 0.6758 | 0.6871 |
| BIOT | ECG only | 20 | 0.2468 | 0.2634 | 0.2656 | 0.2651 | 0.2547 |
