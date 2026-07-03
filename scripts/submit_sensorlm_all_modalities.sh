#!/bin/bash
# Submit SensorLM from-scratch training for EEG_ONLY, ECG_ONLY, EEG_ECG.
#
# Usage: bash scripts/submit_sensorlm_all_modalities.sh
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

for MODALITY in EEG_ONLY ECG_ONLY EEG_ECG; do
    echo "Submitting SensorLM ${MODALITY}"
    sbatch \
        --job-name="sensorlm_${MODALITY}" \
        --output="logs/sensorlm_${MODALITY}_%j.log" \
        scripts/run_sensorlm_modality.slurm ${MODALITY}
done
