# SleepFM on MESA: Complete Results

---

## Section 1: SleepFM Published Encoder (Reference Only)

The published SleepFM encoder was pretrained on data that includes MESA subjects.
There is encoder-level data leakage. These numbers confirm our pipeline reproduces
the paper's reported range (0.70–0.78 macro F1). They are not a fair baseline
and are not included in the main comparison.

**Protocol**: 10-fold cross-validation on all 350 MESA subjects. Not the held-out split.

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM | Per-fold mean |
|---|---|---|---|---|---|---|---|---|
| BAS (EEG+EOG) | 0.7205 | 0.8216 | 0.9326 | 0.3731 | 0.7942 | 0.6522 | 0.8148 | 0.7194 ± 0.0153 |
| BAS+EKG | 0.7237 | 0.8232 | 0.9361 | 0.3829 | 0.7950 | 0.6571 | 0.8172 | 0.7230 ± 0.0129 |
| BAS+EKG+RESP | 0.7269 | 0.8253 | 0.9394 | 0.3916 | 0.7971 | 0.6641 | 0.8252 | 0.7263 ± 0.0118 |
| BAS+EKG+RESP+EMG | 0.7305 | 0.8259 | 0.9399 | 0.4193 | 0.7957 | 0.6528 | 0.8385 | 0.7298 ± 0.0148 |

---

## Section 2: SleepFM From-Scratch - Full Modality Ablation

From-scratch encoder trained on 270 MESA subjects only. No leakage.
Evaluated on the held-out 50-subject test split.

### 2.1 BAS Encoder (EEG+EOG based pretraining)

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| BAS (EEG+EOG) | 0.6509 | 0.7702 | 0.8909 | 0.2203 | 0.7420 | 0.6550 | 0.7563 |
| BAS+EKG | 0.6310 | 0.7610 | 0.8815 | 0.1784 | 0.7453 | 0.6498 | 0.7301 |
| BAS+EKG+RESP | 0.6496 | 0.7755 | 0.9087 | 0.2255 | 0.7687 | 0.6644 | 0.7106 |
| BAS+EKG+RESP+EMG | 0.6601 | 0.7802 | 0.9089 | 0.2794 | 0.7697 | 0.6243 | 0.7583 |

### 2.2 EEG_ONLY Encoder (EEG-only pretraining, no EOG)

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG only | 0.6582 | 0.7757 | 0.8941 | 0.2952 | 0.7653 | 0.6464 | 0.7201 |
| ECG only | 0.3353 | 0.5268 | 0.7566 | 0.0000 | 0.4148 | 0.2188 | 0.3165 |
| EEG+ECG | 0.6529 | 0.7765 | 0.9069 | 0.2411 | 0.7542 | 0.6309 | 0.7415 |

### 2.3 Spectral Encoder (EEG/EOG spectral reconstruction pretraining)

Encoder pretrained on BAS (EEG+EOG) channels via spectral band-power reconstruction,
then fine-tuned end-to-end. ECG_ONLY reflects zero-shot transfer from a non-ECG encoder.

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG only | 0.6971 | 0.8042 | 0.9237 | 0.3628 | 0.7788 | 0.6389 | 0.7811 |
| ECG only | 0.2603 | 0.4216 | 0.6808 | 0.0000 | 0.2645 | 0.1502 | 0.2062 |
| EEG+ECG | 0.6875 | 0.7916 | 0.9133 | 0.3681 | 0.7641 | 0.6195 | 0.7726 |

### 2.4 Combined Encoder (contrastive + spectral pretraining)

Encoder pretrained with combined contrastive + spectral reconstruction (lambda_spectral=0.5) and random-offset window sampling fix, then fine-tuned end-to-end. ECG_ONLY reflects fine-tuned transfer of the combined encoder to ECG-only input.

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG only | 0.6160 | 0.7406 | 0.8658 | 0.1938 | 0.7074 | 0.6257 | 0.6875 |
| ECG only | 0.3380 | 0.5201 | 0.7462 | 0.0000 | 0.4307 | 0.2180 | 0.2949 |
| EEG+ECG | 0.6054 | 0.7412 | 0.8720 | 0.1761 | 0.7165 | 0.5745 | 0.6877 |

---

## Section 3: Main Comparison - All Models on EEG, ECG, EEG+ECG

All results use the same fold_0 held-out test split.
YASA ECG_ONLY = N/A (requires EEG channel, cannot run ECG-only).
YASA EEG+ECG result is identical to EEG_ONLY (ECG input is ignored by YASA).
SleepFM From-Scratch uses the EEG_ONLY encoder for these 3 modalities.

### EEG Only

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| BIOT (fine-tuned) | **0.7237** | 0.8007 | **0.9325** | **0.5327** | 0.7639 | **0.6594** | 0.7301 |
| SleepFM Spectral | 0.6971 | **0.8042** | 0.9237 | 0.3628 | **0.7788** | 0.6389 | **0.7811** |
| LaBraM (fine-tuned) | 0.6835 | 0.7737 | 0.9203 | 0.4257 | 0.7511 | 0.6557 | 0.6647 |
| SleepFM From-Scratch | 0.6582 | 0.7757 | 0.8941 | 0.2952 | 0.7653 | 0.6464 | 0.7201 |
| SensorLM (from scratch) | 0.6264 | 0.7235 | 0.9132 | 0.3549 | 0.6657 | 0.6344 | 0.5639 |
| SleepFM Combined | 0.6160 | 0.7406 | 0.8658 | 0.1938 | 0.7074 | 0.6257 | 0.6875 |
| MOMENT (frozen) | 0.5894 | 0.6745 | 0.8163 | 0.3166 | 0.6900 | 0.6134 | 0.5105 |
| YASA (zero-shot) | 0.4720 | 0.6687 | 0.7843 | 0.1519 | 0.6913 | 0.1795 | 0.5531 |

### ECG Only

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| SleepFM Combined | **0.3380** | 0.5201 | 0.7462 | 0.0000 | 0.4307 | 0.2180 | 0.2949 |
| SleepFM From-Scratch | 0.3353 | **0.5268** | **0.7566** | 0.0000 | 0.4148 | **0.2188** | **0.3165** |
| MOMENT (frozen) | 0.3096 | 0.4191 | 0.6239 | **0.1850** | **0.4467** | 0.1498 | 0.1427 |
| BIOT (fine-tuned) | 0.3086 | 0.4302 | 0.6815 | 0.1239 | 0.3933 | 0.1391 | 0.2051 |
| SensorLM (from scratch) | 0.2821 | 0.3695 | 0.6257 | 0.1330 | 0.3307 | 0.1296 | 0.1914 |
| LaBraM (fine-tuned) | 0.2803 | 0.4092 | 0.6211 | 0.1049 | 0.3920 | 0.1045 | 0.1792 |
| SleepFM Spectral | 0.2603 | 0.4216 | 0.6808 | 0.0000 | 0.2645 | 0.1502 | 0.2062 |
| YASA | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Spectral pretraining used only EEG/EOG (BAS) channels, never ECG. ECG_ONLY result reflects zero-shot transfer from a non-ECG encoder.

### EEG + ECG

| Model | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| BIOT (fine-tuned) | **0.7023** | **0.7975** | **0.9345** | **0.5001** | **0.7724** | 0.5733 | 0.7311 |
| SleepFM Spectral | 0.6875 | 0.7916 | 0.9133 | 0.3681 | 0.7641 | 0.6195 | **0.7726** |
| SleepFM From-Scratch | 0.6529 | 0.7765 | 0.9069 | 0.2411 | 0.7542 | 0.6309 | 0.7415 |
| LaBraM (fine-tuned) | 0.6524 | 0.7566 | 0.9203 | 0.3594 | 0.7431 | **0.6457** | 0.5935 |
| SensorLM (from scratch) | 0.6077 | 0.7062 | 0.8935 | 0.3337 | 0.6891 | 0.6155 | 0.5068 |
| SleepFM Combined | 0.6054 | 0.7412 | 0.8720 | 0.1761 | 0.7165 | 0.5745 | 0.6877 |
| MOMENT (frozen) | 0.5953 | 0.6918 | 0.8395 | 0.3035 | 0.7053 | 0.6089 | 0.5194 |
| YASA (EEG1 only) | 0.4720 | 0.6687 | 0.7843 | 0.1519 | 0.6913 | 0.1795 | 0.5531 |

---

## Section 4: BIOT Full Modality Ablation

BIOT was also evaluated on BAS through BAS+EKG+RESP+EMG configurations.

| Modality | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|
| EEG only | 0.7237 | 0.8007 | 0.9325 | 0.5327 | 0.7639 | 0.6594 | 0.7301 |
| ECG only | 0.3086 | 0.4302 | 0.6815 | 0.1239 | 0.3933 | 0.1391 | 0.2051 |
| EEG+ECG | 0.7023 | 0.7975 | 0.9345 | 0.5001 | 0.7724 | 0.5733 | 0.7311 |
| BAS (EEG+EOG) | 0.7583 | 0.8311 | 0.9476 | 0.5645 | 0.7914 | 0.6645 | 0.8235 |
| BAS+EKG | 0.7492 | 0.8252 | 0.9425 | 0.5388 | 0.7960 | 0.6586 | 0.8102 |
| BAS+EKG+RESP | 0.7418 | 0.8161 | 0.9371 | 0.5248 | 0.7867 | 0.6592 | 0.8010 |
| BAS+EKG+RESP+EMG | 0.7331 | 0.8054 | 0.9252 | 0.5177 | 0.7810 | 0.6502 | 0.7913 |

---

## Section 5: U-Sleep (Training Failure at Test Time)

U-Sleep trained for up to 500 epochs. Val_dice reached 0.88 during training.
At test time all three modalities predicted Wake for every epoch.
Macro F1 = 0.12 equals the majority-class Wake-only baseline.
Results excluded from all comparisons.

| Modality | Macro F1 | Accuracy | Note |
|---|---|---|---|
| EEG only | 0.1202 | 0.4298 | All epochs predicted as Wake |
| ECG only | 0.1202 | 0.4298 | All epochs predicted as Wake |
| EEG+ECG | 0.1202 | 0.4298 | All epochs predicted as Wake |

---

## Section 6: YASA 10-fold CV Reference

Standard YASA configuration with EEG+EOG+EMG, 10-fold CV on 350 subjects.
Not on the held-out split - kept as reference only.

| Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| 0.7304 | 0.83 | 0.93 | 0.43 | 0.82 | 0.63 | 0.84 |

Per-fold: 0.7239, 0.7440, 0.7216, 0.7406, 0.7370, 0.7247, 0.7254, 0.7291, 0.7398, 0.7044
Mean: 0.7291 ± 0.0112

---

## Section 7: Raw Classification Reports

### SleepFM From-Scratch - BAS Encoder

```
===== BAS =====
Macro F1: 0.6509 | Accuracy: 0.7702

              precision    recall  f1-score   support
        Wake       0.88      0.91      0.8909     25980
          N1       0.34      0.16      0.2203      4900
          N2       0.75      0.73      0.7420     19203
          N3       0.60      0.72      0.6550      3867
         REM       0.68      0.83      0.7563      6180

===== BAS_EKG =====
Macro F1: 0.6310 | Accuracy: 0.7610

              precision    recall  f1-score   support
        Wake       0.87      0.89      0.8815     25980
          N1       0.33      0.11      0.1784      4900
          N2       0.74      0.73      0.7453     19203
          N3       0.59      0.69      0.6498      3867
         REM       0.64      0.85      0.7301      6180

===== BAS_EKG_RESP =====
Macro F1: 0.6496 | Accuracy: 0.7755

              precision    recall  f1-score   support
        Wake       0.90      0.90      0.9087     25980
          N1       0.36      0.15      0.2255      4900
          N2       0.76      0.76      0.7687     19203
          N3       0.66      0.67      0.6644      3867
         REM       0.61      0.86      0.7106      6180

===== BAS_EKG_RESP_EMG =====
Macro F1: 0.6601 | Accuracy: 0.7802

              precision    recall  f1-score   support
        Wake       0.88      0.91      0.9089     25980
          N1       0.38      0.21      0.2794      4900
          N2       0.75      0.77      0.7697     19203
          N3       0.68      0.58      0.6243      3867
         REM       0.67      0.84      0.7583      6180
```

### SleepFM From-Scratch - EEG_ONLY Encoder

```
===== EEG_ONLY =====
Macro F1: 0.6582 | Accuracy: 0.7757

              precision    recall  f1-score   support
        Wake       0.88      0.91      0.8941     25980
          N1       0.44      0.21      0.2952      4900
          N2       0.77      0.74      0.7653     19203
          N3       0.60      0.68      0.6464      3867
         REM       0.64      0.82      0.7201      6180

===== ECG_ONLY =====
Macro F1: 0.3353 | Accuracy: 0.5268

              precision    recall  f1-score   support
        Wake       0.68      0.83      0.7566     25980
          N1       0.00      0.00      0.0000      4900
          N2       0.53      0.34      0.4148     19203
          N3       0.17      0.26      0.2188      3867
         REM       0.25      0.40      0.3165      6180

===== EEG+ECG =====
Macro F1: 0.6529 | Accuracy: 0.7765

              precision    recall  f1-score   support
        Wake       0.88      0.91      0.9069     25980
          N1       0.38      0.18      0.2411      4900
          N2       0.75      0.76      0.7542     19203
          N3       0.58      0.69      0.6309      3867
         REM       0.70      0.79      0.7415      6180
```

### BIOT

```
===== EEG_ONLY =====
Macro F1: 0.7237 | Accuracy: 0.8007

              precision    recall  f1-score   support
        Wake     0.9536    0.9124    0.9325     26358
          N1     0.4445    0.6644    0.5327      4908
          N2     0.8479    0.6950    0.7639     19228
          N3     0.5679    0.7861    0.6594      3867
         REM     0.6934    0.7709    0.7301      6189

===== ECG_ONLY =====
Macro F1: 0.3086 | Accuracy: 0.4302

              precision    recall  f1-score   support
        Wake     0.8004    0.5933    0.6815     26358
          N1     0.1176    0.1310    0.1239      4908
          N2     0.4162    0.3728    0.3933     19228
          N3     0.1172    0.1709    0.1391      3867
         REM     0.1526    0.3127    0.2051      6189

===== EEG+ECG =====
Macro F1: 0.7023 | Accuracy: 0.7975

              precision    recall  f1-score   support
        Wake     0.9581    0.9119    0.9345     26358
          N1     0.4196    0.6190    0.5001      4908
          N2     0.7996    0.7470    0.7724     19228
          N3     0.6100    0.5407    0.5733      3867
         REM     0.6967    0.7691    0.7311      6189

===== BAS =====
Macro F1: 0.7583 | Accuracy: 0.8311

              precision    recall  f1-score   support
        Wake     0.9728    0.9236    0.9476     26358
          N1     0.5321    0.6011    0.5645      4908
          N2     0.8406    0.7477    0.7914     19228
          N3     0.5425    0.8575    0.6645      3867
         REM     0.7885    0.8619    0.8235      6189

===== BAS_EKG =====
Macro F1: 0.7492 | Accuracy: 0.8252

              precision    recall  f1-score   support
        Wake     0.9680    0.9182    0.9425     26358
          N1     0.4490    0.6736    0.5388      4908
          N2     0.8278    0.7666    0.7960     19228
          N3     0.6875    0.6320    0.6586      3867
         REM     0.7726    0.8517    0.8102      6189

===== BAS_EKG_RESP =====
Macro F1: 0.7418 | Accuracy: 0.8161

              precision    recall  f1-score   support
        Wake     0.9778    0.8997    0.9371     26358
          N1     0.4319    0.6685    0.5248      4908
          N2     0.8249    0.7519    0.7867     19228
          N3     0.6676    0.6512    0.6592      3867
         REM     0.7354    0.8795    0.8010      6189

===== BAS_EKG_RESP_EMG =====
Macro F1: 0.7331 | Accuracy: 0.8054

              precision    recall  f1-score   support
        Wake     0.9923    0.8666    0.9252     26358
          N1     0.4568    0.5974    0.5177      4908
          N2     0.8382    0.7311    0.7810     19228
          N3     0.5256    0.8521    0.6502      3867
         REM     0.6990    0.9118    0.7913      6189
```

### MOMENT (frozen encoder, linear head)

```
===== EEG_ONLY =====
Macro F1: 0.5894 | Accuracy: 0.6745

              precision    recall  f1-score   support
        Wake     0.8961    0.7495    0.8163     26358
          N1     0.2750    0.3731    0.3166      4908
          N2     0.7564    0.6343    0.6900     19228
          N3     0.5232    0.7411    0.6134      3867
         REM     0.4095    0.6778    0.5105      6189

===== ECG_ONLY =====
Macro F1: 0.3096 | Accuracy: 0.4191

              precision    recall  f1-score   support
        Wake     0.8188    0.5040    0.6239     26358
          N1     0.1250    0.3555    0.1850      4908
          N2     0.4314    0.4632    0.4467     19228
          N3     0.1402    0.1608    0.1498      3867
         REM     0.1549    0.1323    0.1427      6189

===== EEG+ECG =====
Macro F1: 0.5953 | Accuracy: 0.6918

              precision    recall  f1-score   support
        Wake     0.9154    0.7752    0.8395     26358
          N1     0.2790    0.3327    0.3035      4908
          N2     0.7418    0.6724    0.7053     19228
          N3     0.5323    0.7111    0.6089      3867
         REM     0.4240    0.6701    0.5194      6189
```

### LaBraM (EEG-pretrained, full fine-tune)

```
===== EEG_ONLY =====
Macro F1: 0.6835 | Accuracy: 0.7737
Channels: EEG1->FZ, EEG2->OZ, EEG3->C4

              precision    recall  f1-score   support
        Wake     0.9503    0.8920    0.9203     26358
          N1     0.3644    0.5118    0.4257      4908
          N2     0.8261    0.6886    0.7511     19228
          N3     0.5611    0.7887    0.6557      3867
         REM     0.6085    0.7324    0.6647      6189

===== ECG_ONLY =====
Macro F1: 0.2803 | Accuracy: 0.4092
Channels: EKG->T9 (non-EEG channel forced into EEG montage)

              precision    recall  f1-score   support
        Wake     0.6879    0.5661    0.6211     26358
          N1     0.1084    0.1017    0.1049      4908
          N2     0.3964    0.3878    0.3920     19228
          N3     0.0838    0.1386    0.1045      3867
         REM     0.1508    0.2206    0.1792      6189

===== EEG+ECG =====
Macro F1: 0.6524 | Accuracy: 0.7566
Channels: EEG1->FZ, EEG2->OZ, EEG3->C4, EKG->T9

              precision    recall  f1-score   support
        Wake     0.9638    0.8805    0.9203     26358
          N1     0.3123    0.4232    0.3594      4908
          N2     0.7711    0.7171    0.7431     19228
          N3     0.6204    0.6731    0.6457      3867
         REM     0.5339    0.6680    0.5935      6189
```

### SensorLM (ViT-B, trained from scratch on MESA)

```
===== EEG_ONLY =====
Macro F1: 0.6264 | Accuracy: 0.7235

              precision    recall  f1-score   support
        Wake     0.9246    0.9021    0.9132     26358
          N1     0.2783    0.4898    0.3549      4908
          N2     0.8066    0.5667    0.6657     19228
          N3     0.5551    0.7401    0.6344      3867
         REM     0.5137    0.6250    0.5639      6189

===== ECG_ONLY =====
Macro F1: 0.2821 | Accuracy: 0.3695

              precision    recall  f1-score   support
        Wake     0.7948    0.5159    0.6257     26358
          N1     0.0973    0.2101    0.1330      4908
          N2     0.4187    0.2732    0.3307     19228
          N3     0.0911    0.2247    0.1296      3867
         REM     0.1508    0.2621    0.1914      6189

===== EEG+ECG =====
Macro F1: 0.6077 | Accuracy: 0.7062

              precision    recall  f1-score   support
        Wake     0.9508    0.8427    0.8935     26358
          N1     0.2566    0.4770    0.3337      4908
          N2     0.7734    0.6214    0.6891     19228
          N3     0.5222    0.7494    0.6155      3867
         REM     0.4754    0.5427    0.5068      6189
```

### YASA (zero-shot, held-out split)

```
===== EEG_ONLY =====
Macro F1: 0.4720 | Accuracy: 0.6687
EEG channel: EEG1 only

              precision    recall  f1-score   support
        Wake     0.7611    0.8091    0.7843     26358
          N1     0.2415    0.1108    0.1519      4908
          N2     0.6239    0.7751    0.6913     19228
          N3     0.6768    0.1034    0.1795      3867
         REM     0.5716    0.5358    0.5531      6189

===== ECG_ONLY =====
N/A - YASA requires an EEG channel. Cannot run ECG-only.

===== EEG+ECG =====
Macro F1: 0.4720 | Accuracy: 0.6687
Note: YASA uses EEG1 only. ECG is ignored. Result identical to EEG_ONLY.
```

### SleepFM Spectral Pretrained (EEG/EOG spectral reconstruction, fine-tuned end-to-end)

```
===== EEG_ONLY =====
Macro F1: 0.6971 | Accuracy: 0.8042

              precision    recall  f1-score   support
        Wake     0.9229    0.9244    0.9237     26007
          N1     0.4977    0.2855    0.3628      4943
          N2     0.7832    0.7744    0.7788     19255
          N3     0.5591    0.7453    0.6389      3891
         REM     0.7274    0.8434    0.7811      6189

===== ECG_ONLY =====
Macro F1: 0.2603 | Accuracy: 0.4216
Note: spectral pretraining used only EEG/EOG (BAS) channels, never ECG.

              precision    recall  f1-score   support
        Wake     0.6457    0.7200    0.6808     26007
          N1     0.0000    0.0000    0.0000      4943
          N2     0.4370    0.1897    0.2645     19255
          N3     0.1076    0.2485    0.1502      3891
         REM     0.1489    0.3353    0.2062      6189

===== EEG+ECG =====
Macro F1: 0.6875 | Accuracy: 0.7916

              precision    recall  f1-score   support
        Wake     0.9223    0.9044    0.9133     26007
          N1     0.4877    0.2956    0.3681      4943
          N2     0.7697    0.7586    0.7641     19255
          N3     0.5331    0.7394    0.6195      3891
         REM     0.7089    0.8489    0.7726      6189
```

### SleepFM Combined Pretrained (contrastive + spectral, fine-tuned end-to-end)

```
===== EEG_ONLY =====
Macro F1: 0.6160 | Accuracy: 0.7406

              precision    recall  f1-score   support
        Wake     0.8435    0.8894    0.8658     26007
          N1     0.3621    0.1323    0.1938      4943
          N2     0.7128    0.7022    0.7074     19255
          N3     0.5482    0.7286    0.6257      3891
         REM     0.6513    0.7279    0.6875      6189

    accuracy                         0.7406     60285
   macro avg     0.6236    0.6361    0.6160     60285
weighted avg     0.7235    0.7406    0.7263     60285

===== ECG_ONLY =====
Macro F1: 0.3380 | Accuracy: 0.5201
Note: combined pretraining spectral component used EEG/EOG (BAS) channels only.

              precision    recall  f1-score   support
        Wake     0.7142    0.7813    0.7462     26007
          N1     0.0000    0.0000    0.0000      4943
          N2     0.4721    0.3961    0.4307     19255
          N3     0.1773    0.2827    0.2180      3891
         REM     0.2438    0.3732    0.2949      6189

    accuracy                         0.5201     60285
   macro avg     0.3215    0.3667    0.3380     60285
weighted avg     0.4953    0.5201    0.5038     60285

===== EEG+ECG =====
Macro F1: 0.6054 | Accuracy: 0.7412

              precision    recall  f1-score   support
        Wake     0.8588    0.8856    0.8720     26007
          N1     0.3407    0.1188    0.1761      4943
          N2     0.7136    0.7194    0.7165     19255
          N3     0.5181    0.6448    0.5745      3891
         REM     0.6282    0.7597    0.6877      6189

    accuracy                         0.7412     60285
   macro avg     0.6119    0.6257    0.6054     60285
weighted avg     0.7243    0.7412    0.7272     60285
```

### U-Sleep (training failure at test time)

```
Training completed (val_dice up to 0.88 during training).
At test time all three modalities predicted Wake for every epoch.

===== EEG_ONLY / ECG_ONLY / EEG+ECG =====
Macro F1: 0.1202 | Accuracy: 0.4298 (all three identical)

              precision    recall  f1-score   support
        Wake       0.43      1.00      0.60     30944
          N1       0.00      0.00      0.00      5842
          N2       0.00      0.00      0.00     23257
          N3       0.00      0.00      0.00      4738
         REM       0.00      0.00      0.00      7221

Note: 58 subjects and 72002 epochs vs 49/60550 for other models.
The ut predict evaluation path includes validation subjects.
val_dice during training was not a reliable proxy for test macro F1
on this class-imbalanced dataset.
```
