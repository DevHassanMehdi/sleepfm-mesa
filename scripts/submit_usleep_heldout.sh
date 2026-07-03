#!/bin/bash
# Submit U-Sleep training+eval on the held-out 50-subject test split
# (sleepfm/configs/dataset_split_fromscratch_staging.json, fold_0)
# for the 3 modality configurations used across all other models.
#
# Usage: bash scripts/submit_usleep_heldout.sh
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

for MODALITY in EEG_ONLY ECG_ONLY EEG_ECG; do
    echo "Submitting U-Sleep ${MODALITY} (fold 0, held-out split)"
    sbatch \
        --job-name="usleep_${MODALITY}" \
        --output="logs/usleep_${MODALITY}_%j.log" \
        scripts/train_usleep_fold.slurm 0 ${MODALITY}
done
