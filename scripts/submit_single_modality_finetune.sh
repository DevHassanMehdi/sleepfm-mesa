#!/bin/bash
# Submit EEG_ONLY, ECG_ONLY, EEG_ECG fine-tuning jobs in parallel
cd /users/hamehdi/projects/sleepfm-mesa

echo "Submitting EEG_ONLY..."
sbatch scripts/finetune_EEG_ONLY.slurm

echo "Submitting ECG_ONLY..."
sbatch scripts/finetune_ECG_ONLY.slurm

echo "Submitting EEG_ECG..."
sbatch scripts/finetune_EEG_ECG.slurm

echo "All three jobs submitted."
