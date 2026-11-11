#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=check_kaggle_auth
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/check_kaggle_auth_%j.log

echo "=== compute node: $(hostname) ==="
source /scratch/project_2019517/kaggle_venv/bin/activate
unset KAGGLE_USERNAME KAGGLE_KEY
echo "=== kaggle CLI version ==="
kaggle --version
echo "=== authenticated API call (dataset file listing) ==="
kaggle datasets files physionet/physionetchallenge2026data 2>&1 | tail -5
echo "exit code: $?"
