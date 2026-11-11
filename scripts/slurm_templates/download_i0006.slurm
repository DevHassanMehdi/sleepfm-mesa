#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=dl_i0006_moody2026
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/dl_i0006_moody2026_%j.log

echo "START: $(date +%s) host=$(hostname)"
source /scratch/project_2019517/kaggle_venv_py312/bin/activate
unset KAGGLE_USERNAME KAGGLE_KEY
python3 /scratch/project_2019517/claude_arm_build/download_i0006.py
EXIT=$?
echo "END: $(date +%s) exit=$EXIT"
exit $EXIT
