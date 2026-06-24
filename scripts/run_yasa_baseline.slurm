#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=yasa_baseline
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/yasa_baseline_%j.log

# Usage: sbatch run_yasa_baseline.slurm <MODALITY>
# MODALITY one of: EEG_EOG, EEG_ONLY, ECG_ONLY (skipped, not supported by YASA)
MODALITY=${1:?Usage: sbatch run_yasa_baseline.slurm <MODALITY>}

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

python scripts/run_yasa_baseline.py --modality ${MODALITY}

echo "========== DONE (${MODALITY}) =========="
