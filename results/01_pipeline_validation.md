# Pipeline Validation

**Finding:** A label-expansion bug capped every full-cohort score near 0.53. Fixing it restored the published 0.70-0.78 range.

- 1592 of 1944 subjects had un-expanded, row-per-event labels, not row-per-epoch.
- Fixing it lifted full-cohort macro F1 from about 0.53 to 0.74-0.75.
- fold5_v1 is the standard split, it has a tighter SEM than fold10_v1 despite fewer folds.

**Setup:** Full MESA cohort, 2056 subjects after the label fix, fold5_v1 standard. Patience not relevant to this file.

**Legend:** 🟩 best in column, bold value. Lower is best for Std and SEM.

Full details: [PIPELINE_VALIDATION_FINDINGS.md](../PIPELINE_VALIDATION_FINDINGS.md).

## Label bug, before and after

| State | Cohort | Macro F1 |
|---|---|---|
| OLD, corrupted labels | 1944 subjects | ~0.53-0.54 |
| **NEW, fixed labels (mean, all 20 runs)** | 2056 subjects | **0.7418-0.7518 (mean 0.7459)** 🟩 |
| SleepFM published or leaky range | n/a | 0.70-0.78 |

### Published checkpoint, fixed labels, by modality

| Modality | Macro F1 (mean ± std, 5 folds) |
|---|---|
| BAS+EKG+RESP+EMG (full) | **0.7518 ± 0.0022** 🟩 |
| BAS+EKG+RESP | 0.7476 ± 0.0022 |
| BAS+EKG | 0.7424 ± 0.0038 |
| BAS | 0.7418 ± 0.0022 |

## 350-subject label check

| Rule | Subjects fine | Subjects bad |
|---|---|---|
| Lenient (confirmed) | **348 of 350** 🟩 | 2 |
| Strict (exact row match) | 335 of 350 | 15 |

## 5-fold vs 10-fold comparison

| Modality | Split | Folds | Mean Macro F1 | Std | SEM |
|---|---|---|---|---|---|
| EEG_ECG | fold10_v1 | 10 | **0.5187** 🟩 | 0.0182 | 0.0058 |
| EEG_ONLY | fold5_v1 | 5 | 0.5162 | 0.0080 | 0.0036 |
| EEG_ECG | fold5_v1 | 5 | 0.5162 | 0.0090 | **0.0040** |
| EEG_ONLY | fold10_v1 | 10 | 0.5103 | 0.0215 | 0.0068 |
| ECG_ONLY | fold10_v1 | 10 | 0.4006 | 0.0168 | 0.0053 |
| ECG_ONLY | fold5_v1 | 5 | 0.3943 | **0.0109** | 0.0049 |

fold5_v1 recommended, tighter SEM despite half the folds, because its per-fold test sets are larger (388 vs 194 subjects).

## Historical reproductions

| Experiment | Macro F1 | Historical reference |
|---|---|---|
| **Fresh encoder, reproduced today** | **0.6592** 🟩 | 0.6582 |
| Pipeline-regression check, same encoder | 0.6482 | 0.6582 |

### Per-fold macro F1

**EEG_ONLY**

| Split | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| fold5_v1 | 0.5109 | 0.5227 | 0.5222 | 0.5030 | 0.5222 | | | | | |
| fold10_v1 | 0.5145 | 0.4819 | 0.5096 | 0.5332 | 0.5058 | 0.5277 | 0.4691 | 0.4958 | 0.5349 | 0.5309 |

**ECG_ONLY**

| Split | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| fold5_v1 | 0.4036 | 0.3776 | 0.3998 | 0.4051 | 0.3853 | | | | | |
| fold10_v1 | 0.4312 | 0.3764 | 0.4008 | 0.3881 | 0.3974 | 0.4080 | 0.3860 | 0.4251 | 0.4070 | 0.3861 |

**EEG_ECG**

| Split | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| fold5_v1 | 0.5159 | 0.5104 | 0.5067 | 0.5147 | 0.5331 | | | | | |
| fold10_v1 | 0.5218 | 0.4833 | 0.5247 | 0.5388 | 0.5040 | 0.5342 | 0.4966 | 0.5245 | 0.5436 | 0.5156 |

These three are the corrupted-label, 1944-subject fold5/fold10 comparison runs.

**Published checkpoint, fixed labels, fold5_v1**

| Modality | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| BAS+EKG+RESP+EMG (full) | 0.753037 | 0.748314 | 0.754298 | 0.753054 | 0.750485 |
| BAS+EKG+RESP | 0.749565 | 0.746742 | 0.743615 | 0.749563 | 0.748419 |
| BAS+EKG | 0.740103 | 0.745539 | 0.736050 | 0.744698 | 0.745641 |
| BAS | 0.739065 | 0.741539 | 0.741831 | 0.745762 | 0.740803 |

**Published checkpoint, corrupted labels, fold5_v1 (OLD, before reference)**

| Modality | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| BAS_EKG_RESP | 0.5372 | 0.5367 | 0.5386 | 0.5217 | 0.5573 |
| BAS_EKG | 0.5294 | 0.5449 | 0.5166 | 0.5449 | 0.5607 |
| BAS | 0.5232 | 0.5363 | 0.5284 | 0.5429 | 0.5386 |
