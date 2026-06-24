#!/bin/bash
# Submit U-Sleep training+eval for all 10 folds of the SleepFM MESA split.
# Each fold is an independent sbatch job (scripts/train_usleep_fold.slurm).
set -e
cd /users/hamehdi/projects/sleepfm-mesa

for FOLD in $(seq 0 9); do
    echo "Submitting fold ${FOLD}"
    sbatch scripts/train_usleep_fold.slurm ${FOLD}
done
