#!/bin/bash
# Submit LaBraM fine-tuning+eval jobs for all 7 modality configurations.
#
# Non-EEG channels use forced placeholder 10-20 electrode mappings for
# research purposes (see scripts/labram_dataset.py for the mapping).
#
# Usage: bash scripts/submit_labram_all_modalities.sh
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

MODALITIES="EEG_ONLY ECG_ONLY EEG_ECG"

for MODALITY in $MODALITIES; do
    echo "Submitting LaBraM ${MODALITY}"
    sbatch \
        --job-name="labram_${MODALITY}" \
        --output="/users/hamehdi/projects/sleepfm-mesa/logs/labram_${MODALITY}_%j.log" \
        scripts/run_labram_modality.slurm ${MODALITY}
done
