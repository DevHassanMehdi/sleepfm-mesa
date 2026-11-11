# From-Scratch ECG-Only Ablation: Summary

Full details, per-fold numbers, and provenance are in
`results/detailed/ECG_ABLATION_DETAILED.md`. This file summarizes that
data; every number below is taken from it.

## Configuration

| Setting | Value |
|---|---|
| Cohort | 2056 subjects (full MESA cohort) |
| Fold scheme | `fold5_v1`, subject-level 5-fold (fine-tuning only; pretraining uses a single train/val split, not folded) |
| Fine-tuning patience | 50 (hardcoded in `finetune_sleep_staging.py` at the time these runs were made) |
| Pretraining epochs | 100 (configured; the old 350-subject encoder's surviving log only shows 67 epochs actually logged) |
| Batch size | 128 |
| Optimizer | SGD, momentum=0.9 |
| Learning rate | 0.001 |
| Weight decay | Old config: 0.0. Fixed config: 0.0001 |
| LR schedule | Old config: cuts the learning rate 10x every 2 epochs (`lr_step_period=2`, `gamma=0.1`). Fixed config: constant (`lr_step_period=null`) |
| Chunk length | 5-minute windows (`sampling_duration=5`, `sampling_freq=128`Hz, `patch_size=640`) |

## Results by pretraining setup

| Pretraining setup | Pretraining subjects | Signal groups in pretraining | ECG-only macro F1 | EEG-only macro F1 | EEG+ECG macro F1 |
|---|---|---|---|---|---|
| Arm C | 2056 | EKG, RESP | **0.5602 ± 0.0061** | -- | -- |
| seed123 | 2056 | EEG_ONLY, EKG (seed 123) | 0.5596 ± 0.0055 | **0.7489 ± 0.0037** | 0.7503 ± 0.0050 |
| Reference | 2056 | EEG_ONLY, EKG | 0.5575 ± 0.0059 | 0.7451 | 0.7435 |
| Arm A | 2056 | EEG_ONLY, RESP, EKG, EMG | 0.5405 ± 0.0060 | 0.7468 ± 0.0035 | **0.7505 ± 0.0032** |
| Arm B | 2056 | RESP, EKG, EMG | 0.5329 ± 0.0057 | -- | -- |
| Arm E (300-subject retrain, fixed settings) | 300 | EEG_ONLY, RESP, EKG, EMG | 0.5231 ± 0.0064 | 0.7303 ± 0.0019 | -- |
| Arm D | 2056 | EKG, EMG | 0.5173 ± 0.0084 | -- | -- |
| Old 350-subject encoder (rerun on corrected cohort) | 300 | EEG_ONLY, RESP, EKG, EMG (old, buggy hyperparams) | 0.3899 ± 0.0045 | 0.6954 ± 0.0027 | -- |

Rows are sorted by ECG-only macro F1, highest first. "--" means that
fine-tuning combination was not run for that pretraining setup (Arm B/C/D
have no EEG in pretraining, so EEG-only/EEG+ECG fine-tuning isn't
applicable; the old encoder and Arm E were only fine-tuned on ECG-only
and EEG-only, not EEG+ECG). Reference's EEG-only and EEG+ECG means have
no std reported in the source file.

The 350-subject paper reported 0.3353 for From-Scratch ECG-only on its
own 50-subject test set. It is not in the table because the test set
differs.

## Discussion

From-Scratch's ECG-only score is 0.56 on the full cohort. The 350-subject
paper reported 0.34, on a 50-subject test set. To find out why it rose,
we reran the old encoder on the full-cohort folds. It scores 0.39, so
that is the fair starting point. Retraining on the same 300 subjects with
a fixed learning rate raises it to 0.52. Training on all 2056 subjects
raises it to 0.54. Narrowing pretraining to EEG and ECG raises it to
0.56. About 80% of the gain from 0.39 to 0.56 comes from fixing the
training schedule. The old config cut the learning rate by 10 times
every 2 epochs and had no weight decay. We changed both together, so we
cannot say how much each one did.

EEG in pretraining is not needed. Encoders that never saw EEG score
between 0.52 and 0.56. The partner signal matters a little. Chin EMG
gives the lowest score. A second pretraining seed matches the main run
within 0.002 on ECG-only. Each setup was pretrained once, so differences
of 0.02 to 0.04 are not reliable.

Labels do not explain the gap. Pretraining does not use labels, and
every run was tested on the same corrected labels. A check of the
350-subject labels found 348 of 350 were fine.
