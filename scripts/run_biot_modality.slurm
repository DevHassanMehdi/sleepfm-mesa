#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=biot_modality
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/biot_modality_%j.log

# Usage: sbatch run_biot_modality.slurm <MODALITY>
# MODALITY one of: EEG_ONLY, ECG_ONLY, EEG_ECG, BAS, BAS_EKG, BAS_EKG_RESP, BAS_EKG_RESP_EMG
set -e
MODALITY=${1:?Usage: sbatch run_biot_modality.slurm <MODALITY>}

cd /users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

echo "===== Fine-tuning BIOT (${MODALITY}) ====="
python scripts/finetune_biot.py --modality ${MODALITY}

echo "===== Evaluating BIOT (${MODALITY}) ====="
python scripts/evaluate_biot.py --modality ${MODALITY}

echo "========== DONE (${MODALITY}) =========="
