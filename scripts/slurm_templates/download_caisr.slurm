#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=dl_caisr_i0006
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/dl_caisr_i0006_%j.log

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
source /scratch/project_2019517/kaggle_venv_py312/bin/activate
unset KAGGLE_USERNAME KAGGLE_KEY

echo "START: $(date +%s)"
mkdir -p /scratch/project_2019517/moody2026_pilot/raw_large_caisr
chmod 700 /scratch/project_2019517/moody2026_pilot/raw_large_caisr
python3 /scratch/project_2019517/claude_arm_build/download_caisr.py
EXIT=$?
echo "END: $(date +%s) exit=$EXIT"
exit $EXIT
