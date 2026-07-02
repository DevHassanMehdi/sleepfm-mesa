#!/bin/bash
# Submit BIOT fine-tuning+eval jobs for all 7 modality configurations.
#
# Usage: bash scripts/submit_biot_all_modalities.sh
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

MODALITIES="EEG_ONLY ECG_ONLY EEG_ECG"

for MODALITY in $MODALITIES; do
    echo "Submitting BIOT ${MODALITY}"
    sbatch \
        --job-name="biot_${MODALITY}" \
        --output="/users/hamehdi/projects/sleepfm-mesa/logs/biot_${MODALITY}_%j.log" \
        scripts/run_biot_modality.slurm ${MODALITY}
done
