#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=check_kaggle_auth_v2
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/check_kaggle_auth_v2_%j.log

echo "=== compute node: $(hostname) ==="
source /scratch/project_2019517/kaggle_venv_py312/bin/activate
unset KAGGLE_USERNAME KAGGLE_KEY KAGGLE_API_TOKEN
echo "=== kaggle CLI version ==="
kaggle --version
echo "=== authenticated API call (dataset file listing), file count only ==="
kaggle datasets files physionet/physionetchallenge2026data > /tmp/kaggle_filelist_$$.txt 2>&1
EXIT=$?
echo "exit code: $EXIT"
echo "line count: $(wc -l < /tmp/kaggle_filelist_$$.txt)"
echo "first line: $(head -1 /tmp/kaggle_filelist_$$.txt)"
rm -f /tmp/kaggle_filelist_$$.txt
