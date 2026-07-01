# SleepFM on MESA: All Results

---

## Published Encoder (10-fold CV)

| Modality | Channels | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| BAS | EEG1-3, EOG-L, EOG-R | 0.7205 | 0.8216 | 0.9326 | 0.3731 | 0.7942 | 0.6522 | 0.8148 |
| BAS+EKG | EEG1-3, EOG-L/R, EKG | 0.7237 | 0.8232 | 0.9361 | 0.3829 | 0.7950 | 0.6571 | 0.8172 |
| BAS+EKG+RESP | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor | 0.7269 | 0.8253 | 0.9394 | 0.3916 | 0.7971 | 0.6641 | 0.8252 |
| BAS+EKG+RESP+EMG | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor, EMG, Leg, Pleth | 0.7305 | 0.8259 | 0.9399 | 0.4193 | 0.7957 | 0.6528 | 0.8385 |

---

## From-Scratch Encoder — BAS (Held-out 50 Test)

| Modality | Channels | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| BAS | EEG1-3, EOG-L, EOG-R | 0.6509 | 0.7702 | 0.8909 | 0.2203 | 0.7420 | 0.6550 | 0.7563 |
| BAS_EKG | EEG1-3, EOG-L/R, EKG | 0.6310 | 0.7610 | 0.8815 | 0.1784 | 0.7453 | 0.6498 | 0.7301 |
| BAS_EKG_RESP | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor | 0.6496 | 0.7755 | 0.9087 | 0.2255 | 0.7687 | 0.6644 | 0.7106 |
| BAS_EKG_RESP_EMG | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor, EMG, Leg, Pleth | 0.6601 | 0.7802 | 0.9089 | 0.2794 | 0.7697 | 0.6243 | 0.7583 |

---

## From-Scratch Encoder — EEG_ONLY (Held-out 50 Test)

| Modality | Channels | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| EEG_ONLY | EEG1, EEG2, EEG3 | 0.6582 | 0.7757 | 0.8941 | 0.2952 | 0.7653 | 0.6464 | 0.7201 |
| ECG_ONLY | EKG | 0.3353 | 0.5268 | 0.7566 | 0.0000 | 0.4148 | 0.2188 | 0.3165 |
| EEG_ECG | EEG1, EEG2, EEG3, EKG | 0.6529 | 0.7765 | 0.9069 | 0.2411 | 0.7542 | 0.6309 | 0.7415 |

---

## BIOT (EEG-SHHS+PREST, End-to-End Fine-tuned, Held-out 50 Test)

| Modality | Channels | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| EEG_ONLY | EEG1, EEG2, EEG3 | 0.7237 | 0.8007 | 0.9325 | 0.5327 | 0.7639 | 0.6594 | 0.7301 |
| ECG_ONLY | EKG | 0.3086 | 0.4302 | 0.6815 | 0.1239 | 0.3933 | 0.1391 | 0.2051 |
| EEG_ECG | EEG1, EEG2, EEG3, EKG | 0.7023 | 0.7975 | 0.9345 | 0.5001 | 0.7724 | 0.5733 | 0.7311 |
| BAS | EEG1-3, EOG-L, EOG-R | 0.7583 | 0.8311 | 0.9476 | 0.5645 | 0.7914 | 0.6645 | 0.8235 |
| BAS_EKG | EEG1-3, EOG-L/R, EKG | 0.7492 | 0.8252 | 0.9425 | 0.5388 | 0.7960 | 0.6586 | 0.8102 |
| BAS_EKG_RESP | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor | 0.7418 | 0.8161 | 0.9371 | 0.5248 | 0.7867 | 0.6592 | 0.8010 |
| BAS_EKG_RESP_EMG | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor, EMG, Leg, Pleth | 0.7331 | 0.8054 | 0.9252 | 0.5177 | 0.7810 | 0.6502 | 0.7913 |

---

## Master Comparison Table (All Models, All Modalities)

| Model | Modality | Channels | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|---|
| BIOT | BAS | EEG1-3, EOG-L, EOG-R | 0.7583 | 0.8311 | 0.9476 | 0.5645 | 0.7914 | 0.6645 | 0.8235 |
| Published SleepFM | BAS | EEG1-3, EOG-L, EOG-R | 0.7205 | 0.8216 | 0.9326 | 0.3731 | 0.7942 | 0.6522 | 0.8148 |
| FS SleepFM-BAS | BAS | EEG1-3, EOG-L, EOG-R | 0.6509 | 0.7702 | 0.8909 | 0.2203 | 0.7420 | 0.6550 | 0.7563 |
| BIOT | BAS_EKG | EEG1-3, EOG-L/R, EKG | 0.7492 | 0.8252 | 0.9425 | 0.5388 | 0.7960 | 0.6586 | 0.8102 |
| Published SleepFM | BAS_EKG | EEG1-3, EOG-L/R, EKG | 0.7237 | 0.8232 | 0.9361 | 0.3829 | 0.7950 | 0.6571 | 0.8172 |
| FS SleepFM-BAS | BAS_EKG | EEG1-3, EOG-L/R, EKG | 0.6310 | 0.7610 | 0.8815 | 0.1784 | 0.7453 | 0.6498 | 0.7301 |
| BIOT | BAS_EKG_RESP | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor | 0.7418 | 0.8161 | 0.9371 | 0.5248 | 0.7867 | 0.6592 | 0.8010 |
| Published SleepFM | BAS_EKG_RESP | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor | 0.7269 | 0.8253 | 0.9394 | 0.3916 | 0.7971 | 0.6641 | 0.8252 |
| FS SleepFM-BAS | BAS_EKG_RESP | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor | 0.6496 | 0.7755 | 0.9087 | 0.2255 | 0.7687 | 0.6644 | 0.7106 |
| BIOT | BAS_EKG_RESP_EMG | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor, EMG, Leg, Pleth | 0.7331 | 0.8054 | 0.9252 | 0.5177 | 0.7810 | 0.6502 | 0.7913 |
| Published SleepFM | BAS_EKG_RESP_EMG | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor, EMG, Leg, Pleth | 0.7305 | 0.8259 | 0.9399 | 0.4193 | 0.7957 | 0.6528 | 0.8385 |
| FS SleepFM-BAS | BAS_EKG_RESP_EMG | EEG1-3, EOG-L/R, EKG, Abdo, HR, Snore, SpO2, Therm, Thor, EMG, Leg, Pleth | 0.6601 | 0.7802 | 0.9089 | 0.2794 | 0.7697 | 0.6243 | 0.7583 |
| FS SleepFM-EEG | ECG_ONLY | EKG | 0.3353 | 0.5268 | 0.7566 | 0.0000 | 0.4148 | 0.2188 | 0.3165 |
| BIOT | ECG_ONLY | EKG | 0.3086 | 0.4302 | 0.6815 | 0.1239 | 0.3933 | 0.1391 | 0.2051 |
| BIOT | EEG_ECG | EEG1, EEG2, EEG3, EKG | 0.7023 | 0.7975 | 0.9345 | 0.5001 | 0.7724 | 0.5733 | 0.7311 |
| FS SleepFM-EEG | EEG_ECG | EEG1, EEG2, EEG3, EKG | 0.6529 | 0.7765 | 0.9069 | 0.2411 | 0.7542 | 0.6309 | 0.7415 |
| BIOT | EEG_ONLY | EEG1, EEG2, EEG3 | 0.7237 | 0.8007 | 0.9325 | 0.5327 | 0.7639 | 0.6594 | 0.7301 |
| FS SleepFM-EEG | EEG_ONLY | EEG1, EEG2, EEG3 | 0.6582 | 0.7757 | 0.8941 | 0.2952 | 0.7653 | 0.6464 | 0.7201 |
