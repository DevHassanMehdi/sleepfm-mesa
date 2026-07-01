#!/bin/bash
# Submit LaBraM fine-tuning+eval jobs for all supported modality configurations.
#
# Only EEG_ONLY is supported -- LaBraM is an EEG-only foundation model with
# no defensible channel mapping for MESA's EOG channels (see
# scripts/labram_dataset.py docstring).
#
# Usage: bash scripts/submit_labram_all_modalities.sh
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

MODALITIES="EEG_ONLY"

for MODALITY in $MODALITIES; do
    echo "Submitting LaBraM ${MODALITY}"
    sbatch \
        --job-name="labram_${MODALITY}" \
        --output="/users/hamehdi/projects/sleepfm-mesa/logs/labram_${MODALITY}_%j.log" \
        scripts/run_labram_modality.slurm ${MODALITY}
done
