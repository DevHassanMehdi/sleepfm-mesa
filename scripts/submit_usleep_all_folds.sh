#!/bin/bash
# Submit U-Sleep training+eval for all 10 folds of the SleepFM MESA split,
# for one modality configuration.
#
# Usage: bash scripts/submit_usleep_all_folds.sh <MODALITY>
# MODALITY one of: EEG_EOG, EEG_ONLY, ECG_ONLY, EEG_ECG
set -e
MODALITY=${1:?Usage: bash submit_usleep_all_folds.sh <MODALITY>}
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

for FOLD in $(seq 0 9); do
    echo "Submitting ${MODALITY} fold ${FOLD}"
    sbatch \
        --job-name="usleep_${MODALITY}_f${FOLD}" \
        --output="/users/hamehdi/projects/sleepfm-mesa/logs/usleep_${MODALITY}_fold${FOLD}_%j.log" \
        scripts/train_usleep_fold.slurm ${FOLD} ${MODALITY}
done
