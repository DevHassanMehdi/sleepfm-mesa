#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sensorlm_modality
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/sensorlm_modality_%j.log

# Usage: sbatch run_sensorlm_modality.slurm <MODALITY> [FOLD_KEY]
# MODALITY: EEG_ONLY, ECG_ONLY, or EEG_ECG
#
# Trains SensorLM sensor encoder from scratch (random init) on MESA PSG.
# Resumable: checks for latest.pth and continues automatically on restart.
set -e
MODALITY=${1:?Usage: sbatch run_sensorlm_modality.slurm <MODALITY>}
FOLD_KEY=${2:-fold_0}

cd /users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

LATEST_CKPT="/scratch/project_2019517/sensorlm/checkpoints/${MODALITY}/${FOLD_KEY}/latest.pth"
if [ -f "$LATEST_CKPT" ]; then
    echo "===== Resuming SensorLM training (${MODALITY}) from ${LATEST_CKPT} ====="
else
    echo "===== Starting SensorLM training (${MODALITY}) from scratch ====="
fi
python scripts/finetune_sensorlm.py --modality ${MODALITY} --fold_key ${FOLD_KEY}

echo "===== Evaluating SensorLM (${MODALITY}) ====="
python scripts/evaluate_sensorlm.py --modality ${MODALITY} --fold_key ${FOLD_KEY}

echo "========== DONE (${MODALITY}) =========="
