#!/bin/bash
# Submit MOMENT fine-tuning+eval jobs for all 7 modality configurations.
#
# Usage: bash scripts/submit_moment_all_modalities.sh
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

MODALITIES="EEG_ONLY ECG_ONLY EEG_ECG BAS BAS_EKG BAS_EKG_RESP BAS_EKG_RESP_EMG"

for MODALITY in $MODALITIES; do
    echo "Submitting MOMENT ${MODALITY}"
    sbatch \
        --job-name="moment_${MODALITY}" \
        --output="/users/hamehdi/projects/sleepfm-mesa/logs/moment_${MODALITY}_%j.log" \
        scripts/run_moment_modality.slurm ${MODALITY}
done
