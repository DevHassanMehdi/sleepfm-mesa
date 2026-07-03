#!/bin/bash
# Submit YASA held-out test evaluation for EEG_ONLY and EEG_ECG.
# ECG_ONLY is skipped — YASA requires EEG input (no ECG-only classifier).
#
# Usage: bash scripts/submit_yasa_heldout.sh
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

for MODALITY in EEG_ONLY EEG_ECG; do
    echo "Submitting YASA ${MODALITY}"
    sbatch \
        --job-name="yasa_${MODALITY}" \
        --output="logs/yasa_${MODALITY}_%j.log" \
        scripts/run_yasa_heldout.slurm ${MODALITY}
done
