#!/bin/bash
# Submit YASA baseline for all supported modality configurations.
# ECG_ONLY is intentionally omitted: yasa.SleepStaging requires an EEG
# channel and has no ECG-only mode (run_yasa_baseline.py --modality ECG_ONLY
# would just write a "skipped" results file if you ran it anyway).
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

sbatch \
    --job-name="yasa_EEG_EOG" \
    --output="/users/hamehdi/projects/sleepfm-mesa/logs/yasa_EEG_EOG_%j.log" \
    scripts/run_yasa_baseline.slurm EEG_EOG

sbatch \
    --job-name="yasa_EEG_ONLY" \
    --output="/users/hamehdi/projects/sleepfm-mesa/logs/yasa_EEG_ONLY_%j.log" \
    scripts/run_yasa_baseline.slurm EEG_ONLY
