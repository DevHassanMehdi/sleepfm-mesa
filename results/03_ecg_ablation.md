# ECG Ablation

**Finding:** From-Scratch's ECG-only score rose from 0.39 to 0.56 mostly by fixing the training schedule.

- The old config cut the learning rate 10x every 2 epochs with no weight decay.
- The fixed config is constant learning rate with weight decay 0.0001.
- EEG is not needed in pretraining, encoders that never saw EEG score 0.52 to 0.56 on ECG-only.

**Setup:** Full MESA cohort, 2056 subjects (Arm E and the old encoder use 300), fold5_v1, patience 50.

**Legend:** 🟩 best in column, bold value.

## Results by pretraining setup

| Pretraining setup | Pretraining subjects | Signal groups | ECG-only Macro F1 | EEG-only Macro F1 | EEG+ECG Macro F1 |
|---|---|---|---|---|---|
| Arm C | 2056 | EKG, RESP | **0.5602 ± 0.0061** 🟩 | -- | -- |
| seed123 | 2056 | EEG_ONLY, EKG (seed 123) | 0.5596 ± 0.0055 | **0.7489 ± 0.0037** 🟩 | 0.7503 ± 0.0050 |
| Reference | 2056 | EEG_ONLY, EKG | 0.5575 ± 0.0059 | 0.7451 | 0.7435 |
| Arm A | 2056 | EEG_ONLY, RESP, EKG, EMG | 0.5405 ± 0.0060 | 0.7468 ± 0.0035 | **0.7505 ± 0.0032** 🟩 |
| Arm B | 2056 | RESP, EKG, EMG | 0.5329 ± 0.0057 | -- | -- |
| Arm E (300-subject retrain, fixed settings) | 300 | EEG_ONLY, RESP, EKG, EMG | 0.5231 ± 0.0064 | 0.7303 ± 0.0019 | -- |
| Arm D | 2056 | EKG, EMG | 0.5173 ± 0.0084 | -- | -- |
| Old 350-subject encoder (rerun on corrected cohort) | 300 | EEG_ONLY, RESP, EKG, EMG (old hyperparams) | 0.3899 ± 0.0045 | 0.6954 ± 0.0027 | -- |

"--" means that fine-tuning combination was not run for that pretraining setup. Reference's EEG-only and EEG+ECG means have no std in the source.

### Per-fold macro F1 (ECG-only)

| Setup | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| Arm C | 0.5596 | 0.5669 | 0.5602 | 0.5649 | 0.5493 |
| seed123 | 0.5601 | 0.5686 | 0.5602 | 0.5515 | 0.5575 |
| Reference | 0.5595 | 0.5606 | 0.5601 | 0.5616 | 0.5459 |
| Arm A | 0.5293 | 0.5462 | 0.5421 | 0.5450 | 0.5400 |
| Arm B | 0.5237 | 0.5348 | 0.5295 | 0.5401 | 0.5363 |
| Arm E | 0.5221 | 0.5312 | 0.5220 | 0.5279 | 0.5123 |
| Arm D | 0.5114 | 0.5294 | 0.5215 | 0.5193 | 0.5050 |
| Old 350-subject encoder | 0.3891 | 0.3968 | 0.3864 | 0.3927 | 0.3844 |

### Diff vs. reference mean

| Setup | ECG-only diff | EEG-only diff | EEG+ECG diff |
|---|---|---|---|
| Arm C | +0.0027 | -- | -- |
| seed123 | +0.0020 | +0.0038 | +0.0068 |
| Arm A | -0.0170 | +0.0017 | +0.0070 |
| Arm B | -0.0247 | -- | -- |
| Arm E | -0.0344 | -0.0148 | -- |
| Arm D | -0.0402 | -- | -- |
| Old 350-subject encoder | -0.1677 (-0.168 rounded) | -0.0497 (-0.050 rounded) | -- |

Arm A's ECG-only per-fold diff vs reference ranges -0.0059 to -0.0302, mean -0.0170, below the reference on all 5 folds.

Per-fold accuracy (ECG-only, same fold order): Arm A 0.6867/0.6937/0.6898/0.7015/0.6955, Arm B 0.6821/0.6950/0.6844/0.6946/0.6905, Arm C 0.7077/0.7118/0.7014/0.7113/0.7104, Arm D 0.6733/0.6843/0.6740/0.6837/0.6696.

### Per-fold macro F1 (EEG-only and EEG+ECG, where fine-tuned)

| Setup | Modality | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| Arm A | EEG-only | 0.7444 | 0.7524 | 0.7455 | 0.7491 | 0.7427 |
| Arm A | EEG+ECG | 0.7530 | 0.7521 | 0.7451 | 0.7536 | 0.7485 |
| seed123 | EEG-only | 0.7507 | 0.7538 | 0.7444 | 0.7510 | 0.7448 |
| seed123 | EEG+ECG | 0.7508 | 0.7556 | 0.7427 | 0.7554 | 0.7468 |
| Arm E | EEG-only | 0.7299 | 0.7330 | 0.7304 | 0.7313 | 0.7271 |
| Old 350-subject encoder | EEG-only | 0.6956 | 0.6997 | 0.6928 | 0.6967 | 0.6924 |
