#!/bin/bash
# Submit U-Sleep 10-fold training+eval for all 4 modality configurations.
set -e
cd /users/hamehdi/projects/sleepfm-mesa

for MODALITY in EEG_EOG EEG_ONLY ECG_ONLY EEG_ECG; do
    bash scripts/submit_usleep_all_folds.sh ${MODALITY}
done
