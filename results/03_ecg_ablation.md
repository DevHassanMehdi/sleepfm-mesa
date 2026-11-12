# ECG Ablation

**Finding:** SleepFM From-Scratch's ECG-only score rose from 0.39 to 0.56 mostly by fixing the training schedule.

- The old config cut the learning rate 10x every 2 epochs with no weight decay.
- The fixed config is constant learning rate with weight decay 0.0001.
- EEG is not needed in pretraining, encoders that never saw EEG score 0.52 to 0.56 on ECG-only.

## Names used

Every pretraining setup below was verified against its saved `config.json`/pretraining `.yaml` before naming it; see the per-row notes.

| Name in this file | Plain description |
|---|---|
| Pretrained on EEG + respiration + ECG + EMG (all 2056 subjects) | Verified from the config file `config_pretrain_fromscratch_ablation_A_fullcohort.yaml`: its modality list is EEG, respiration, ECG, and EMG; 2056 subjects |
| Pretrained on respiration + ECG + EMG, no EEG | Verified from the config file `config_pretrain_fromscratch_ablation_B_fullcohort.yaml`: its modality list is respiration, ECG, and EMG, with no EEG; 2056 subjects |
| Pretrained on ECG + respiration, no EEG | Verified from the config file `config_pretrain_fromscratch_ablation_C_fullcohort.yaml`: its modality list is ECG and respiration only, no EEG; 2056 subjects |
| Pretrained on ECG + EMG, no EEG | Verified from the config file `config_pretrain_fromscratch_ablation_D_fullcohort.yaml`: its modality list is ECG and EMG only, no EEG; 2056 subjects |
| Same 300 subjects as the old encoder, fixed training settings, EEG + respiration + ECG + EMG | Verified from the config file `config_pretrain_fromscratch_ablation_E_300subj.yaml`: same modality list as the all-signals run above (EEG, respiration, ECG, EMG), 300 subjects, and the fixed learning-rate schedule (no step decay, weight decay 0.0001) |
| Old 350-subject encoder with the old training settings, rerun on the corrected cohort | Verified from the checkpoint's own saved config: the old, buggy learning-rate schedule (learning rate cut by 10x every 2 epochs, no weight decay), 300 subjects |
| Pretrained on EEG + ECG (main benchmark run) | Verified from the config file `config_pretrain_fromscratch_eegonly_fullcohort.yaml`: its modality list is EEG and ECG only, random seed 42, 2056 subjects |
| Same as the main run, second random seed | Verified: same config and modality scope as the main benchmark run, launched with a second-seed command-line override; the config file's own saved seed field is unchanged, the override is a launch-time flag |
| EEG, 3 channels | All three EEG leads used together |
| ECG only | The single ECG lead alone, no EEG |
| EEG + ECG | The three EEG leads plus one ECG lead, together |

All eight names above were checked against their configs; none disagreed with the expected mapping.

**Setup:** Full MESA cohort, 2056 subjects ("Same 300 subjects..." and "Old 350-subject encoder..." use 300), fold5_v1, early-stopping patience 50.

**Legend:** 🟩 best in column, bold value.

## Main results

| Pretraining setup | Pretraining subjects | ECG-only Macro F1 | EEG, 3 channels Macro F1 | EEG + ECG Macro F1 |
|---|---|---|---|---|
| Pretrained on ECG + respiration, no EEG | 2056 | **0.5602 ± 0.0061** 🟩 | -- | -- |
| Same as the main run, second random seed | 2056 | 0.5596 ± 0.0055 | **0.7489 ± 0.0037** 🟩 | 0.7503 ± 0.0050 |
| Pretrained on EEG + ECG (main benchmark run) | 2056 | 0.5575 ± 0.0059 | 0.7451 | 0.7435 |
| Pretrained on EEG + respiration + ECG + EMG (all 2056 subjects) | 2056 | 0.5405 ± 0.0060 | 0.7468 ± 0.0035 | **0.7505 ± 0.0032** 🟩 |
| Pretrained on respiration + ECG + EMG, no EEG | 2056 | 0.5329 ± 0.0057 | -- | -- |
| Same 300 subjects as the old encoder, fixed training settings, EEG + respiration + ECG + EMG | 300 | 0.5231 ± 0.0064 | 0.7303 ± 0.0019 | -- |
| Pretrained on ECG + EMG, no EEG | 2056 | 0.5173 ± 0.0084 | -- | -- |
| Old 350-subject encoder with the old training settings, rerun on the corrected cohort | 300 | 0.3899 ± 0.0045 | 0.6954 ± 0.0027 | -- |

"--" means that fine-tuning combination was not run for that pretraining setup. "Pretrained on EEG + ECG (main benchmark run)"'s EEG, 3-channel and EEG+ECG means have no std in the source.

## Detail tables

### Per-fold macro F1 (ECG-only)

| Pretraining setup | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| Pretrained on ECG + respiration, no EEG | 0.5596 | 0.5669 | 0.5602 | 0.5649 | 0.5493 |
| Same as the main run, second random seed | 0.5601 | 0.5686 | 0.5602 | 0.5515 | 0.5575 |
| Pretrained on EEG + ECG (main benchmark run) | 0.5595 | 0.5606 | 0.5601 | 0.5616 | 0.5459 |
| Pretrained on EEG + respiration + ECG + EMG (all 2056 subjects) | 0.5293 | 0.5462 | 0.5421 | 0.5450 | 0.5400 |
| Pretrained on respiration + ECG + EMG, no EEG | 0.5237 | 0.5348 | 0.5295 | 0.5401 | 0.5363 |
| Same 300 subjects as the old encoder, fixed training settings, EEG + respiration + ECG + EMG | 0.5221 | 0.5312 | 0.5220 | 0.5279 | 0.5123 |
| Pretrained on ECG + EMG, no EEG | 0.5114 | 0.5294 | 0.5215 | 0.5193 | 0.5050 |
| Old 350-subject encoder with the old training settings, rerun on the corrected cohort | 0.3891 | 0.3968 | 0.3864 | 0.3927 | 0.3844 |

### Diff vs. main-benchmark-run mean

| Pretraining setup | ECG-only diff | EEG, 3-channel diff | EEG + ECG diff |
|---|---|---|---|
| Pretrained on ECG + respiration, no EEG | +0.0027 | -- | -- |
| Same as the main run, second random seed | +0.0020 | +0.0038 | +0.0068 |
| Pretrained on EEG + respiration + ECG + EMG (all 2056 subjects) | -0.0170 | +0.0017 | +0.0070 |
| Pretrained on respiration + ECG + EMG, no EEG | -0.0247 | -- | -- |
| Same 300 subjects as the old encoder, fixed training settings, EEG + respiration + ECG + EMG | -0.0344 | -0.0148 | -- |
| Pretrained on ECG + EMG, no EEG | -0.0402 | -- | -- |
| Old 350-subject encoder with the old training settings, rerun on the corrected cohort | -0.1677 (-0.168 rounded) | -0.0497 (-0.050 rounded) | -- |

"Pretrained on EEG + respiration + ECG + EMG (all 2056 subjects)"'s ECG-only per-fold diff vs the main benchmark run ranges -0.0059 to -0.0302, mean -0.0170, below the main benchmark run on all 5 folds.

Per-fold accuracy (ECG-only, same fold order): "Pretrained on EEG + respiration + ECG + EMG (all 2056 subjects)" 0.6867/0.6937/0.6898/0.7015/0.6955, "Pretrained on respiration + ECG + EMG, no EEG" 0.6821/0.6950/0.6844/0.6946/0.6905, "Pretrained on ECG + respiration, no EEG" 0.7077/0.7118/0.7014/0.7113/0.7104, "Pretrained on ECG + EMG, no EEG" 0.6733/0.6843/0.6740/0.6837/0.6696.

### Per-fold macro F1 (EEG, 3 channels and EEG + ECG, where fine-tuned)

| Pretraining setup | Condition | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| Pretrained on EEG + respiration + ECG + EMG (all 2056 subjects) | EEG, 3 channels | 0.7444 | 0.7524 | 0.7455 | 0.7491 | 0.7427 |
| Pretrained on EEG + respiration + ECG + EMG (all 2056 subjects) | EEG + ECG | 0.7530 | 0.7521 | 0.7451 | 0.7536 | 0.7485 |
| Same as the main run, second random seed | EEG, 3 channels | 0.7507 | 0.7538 | 0.7444 | 0.7510 | 0.7448 |
| Same as the main run, second random seed | EEG + ECG | 0.7508 | 0.7556 | 0.7427 | 0.7554 | 0.7468 |
| Same 300 subjects as the old encoder, fixed training settings, EEG + respiration + ECG + EMG | EEG, 3 channels | 0.7299 | 0.7330 | 0.7304 | 0.7313 | 0.7271 |
| Old 350-subject encoder with the old training settings, rerun on the corrected cohort | EEG, 3 channels | 0.6956 | 0.6997 | 0.6928 | 0.6967 | 0.6924 |
