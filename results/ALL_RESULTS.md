# SleepFM on MESA: All Results
**Project:** AI4HOPE, University of Turku

---

## Experiment Overview

| Experiment | Encoder | Protocol | Split |
|---|---|---|---|
| Published encoder modality ablation | Published SleepFM | 10-fold CV | dataset_split_10fold.json |
| From-scratch BAS encoder modality ablation | From-scratch on 270 MESA subjects | Held-out 50 test | fromscratch_heldout_flat.json |
| EEG_ONLY encoder single/paired modality | From-scratch EEG_ONLY on 270 MESA subjects | Held-out 50 test | dataset_split_fromscratch_staging.json |

---

## Table 1: Published Encoder — 10-fold CV

| Modality | Channels | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| BAS | EEG1-3 + EOG-L/R | 0.7205 | 0.8216 | 0.9326 | 0.3731 | 0.7942 | 0.6522 | 0.8148 |
| BAS+EKG | + 1 ECG channel | 0.7237 | 0.8232 | 0.9361 | 0.3829 | 0.7950 | 0.6571 | 0.8172 |
| BAS+EKG+RESP | + 6 respiratory channels | 0.7269 | 0.8253 | 0.9394 | 0.3916 | 0.7971 | 0.6641 | 0.8252 |
| BAS+EKG+RESP+EMG | + 3 EMG channels | 0.7305 | 0.8259 | 0.9399 | 0.4193 | 0.7957 | 0.6528 | 0.8385 |

---

## Table 2: From-Scratch Encoder — Held-out 50 Test

| Modality | Channels | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| BAS | EEG1-3 + EOG-L/R | 0.6509 | 0.7702 | 0.8909 | 0.2203 | 0.7420 | 0.6550 | 0.7563 |
| BAS_EKG | + 1 ECG channel | 0.6310 | 0.7610 | 0.8815 | 0.1784 | 0.7453 | 0.6498 | 0.7301 |
| BAS_EKG_RESP | + 6 respiratory channels | 0.6496 | 0.7755 | 0.9087 | 0.2255 | 0.7687 | 0.6644 | 0.7106 |
| BAS_EKG_RESP_EMG | + 3 EMG channels | 0.6601 | 0.7802 | 0.9089 | 0.2794 | 0.7697 | 0.6243 | 0.7583 |

---

## Table 3: From-Scratch Encoder (EEG_ONLY) — Held-out 50 Test

| Modality | Channels | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| EEG only | EEG1, EEG2, EEG3 | 0.6582 | 0.7757 | 0.8941 | 0.2952 | 0.7653 | 0.6464 | 0.7201 |
| ECG only | 1 ECG channel | 0.3353 | 0.5268 | 0.7566 | 0.0000 | 0.4148 | 0.2188 | 0.3165 |
| EEG+ECG | EEG1-3 + 1 ECG channel | 0.6529 | 0.7765 | 0.9069 | 0.2411 | 0.7542 | 0.6309 | 0.7415 |

---

## Raw Classification Reports

Published encoder (`results/PUB_SleepFM_results.txt`) does not contain full sklearn classification reports (precision/recall/support per class), only the per-class F1 summary used in Table 1 above. No raw report is available for those four rows.

**From-Scratch BAS — BAS**
```
              precision    recall  f1-score   support

        Wake       0.88      0.91      0.8909     25980
          N1       0.34      0.16      0.2203      4900
          N2       0.75      0.73      0.7420     19203
          N3       0.60      0.72      0.6550      3867
         REM       0.68      0.83      0.7563      6180

    accuracy                           0.77     60130
   macro avg       0.65      0.67      0.65     60130
weighted avg       0.75      0.77      0.76     60130
```

**From-Scratch BAS — BAS_EKG**
```
              precision    recall  f1-score   support

        Wake       0.87      0.89      0.8815     25980
          N1       0.33      0.11      0.1784      4900
          N2       0.74      0.73      0.7453     19203
          N3       0.59      0.69      0.6498      3867
         REM       0.64      0.85      0.7301      6180

    accuracy                           0.76     60130
   macro avg       0.63      0.66      0.63     60130
weighted avg       0.74      0.76      0.75     60130
```

**From-Scratch BAS — BAS_EKG_RESP**
```
              precision    recall  f1-score   support

        Wake       0.90      0.90      0.9087     25980
          N1       0.36      0.15      0.2255      4900
          N2       0.76      0.76      0.7687     19203
          N3       0.66      0.67      0.6644      3867
         REM       0.61      0.86      0.7106      6180

    accuracy                           0.78     60130
   macro avg       0.66      0.67      0.65     60130
weighted avg       0.76      0.78      0.76     60130
```

**From-Scratch BAS — BAS_EKG_RESP_EMG**
```
              precision    recall  f1-score   support

        Wake       0.88      0.91      0.9089     25980
          N1       0.38      0.21      0.2794      4900
          N2       0.75      0.77      0.7697     19203
          N3       0.68      0.58      0.6243      3867
         REM       0.67      0.84      0.7583      6180

    accuracy                           0.78     60130
   macro avg       0.67      0.66      0.66     60130
weighted avg       0.77      0.78      0.77     60130
```

**EEG_ONLY encoder — EEG only**
```
              precision    recall  f1-score   support

        Wake       0.88      0.91      0.8941     25980
          N1       0.44      0.21      0.2952      4900
          N2       0.77      0.74      0.7653     19203
          N3       0.60      0.68      0.6464      3867
         REM       0.64      0.82      0.7201      6180

    accuracy                           0.78     60130
   macro avg       0.67      0.67      0.66     60130
weighted avg       0.77      0.78      0.77     60130
```

**EEG_ONLY encoder — ECG only**
```
              precision    recall  f1-score   support

        Wake       0.68      0.83      0.7566     25980
          N1       0.00      0.00      0.0000      4900
          N2       0.53      0.34      0.4148     19203
          N3       0.17      0.26      0.2188      3867
         REM       0.25      0.40      0.3165      6180

    accuracy                           0.53     60130
   macro avg       0.33      0.37      0.34     60130
weighted avg       0.50      0.53      0.50     60130
```

**EEG_ONLY encoder — EEG+ECG**
```
              precision    recall  f1-score   support

        Wake       0.88      0.91      0.9069     25980
          N1       0.38      0.18      0.2411      4900
          N2       0.75      0.76      0.7542     19203
          N3       0.58      0.69      0.6309      3867
         REM       0.70      0.79      0.7415      6180

    accuracy                           0.78     60130
   macro avg       0.66      0.67      0.65     60130
weighted avg       0.76      0.78      0.76     60130
```

**YASA Baseline — EEG+EOG+EMG (all folds combined)**
```
              precision    recall  f1-score   support

        Wake       0.94      0.93      0.93    191677
          N1       0.44      0.43      0.43     33749
          N2       0.80      0.84      0.82    143770
          N3       0.75      0.54      0.63     26517
         REM       0.81      0.87      0.84     45307

    accuracy                           0.83    441020
   macro avg       0.75      0.72      0.73    441020
weighted avg       0.83      0.83      0.83    441020
```

---