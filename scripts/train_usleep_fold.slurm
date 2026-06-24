#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=usleep_fold
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/usleep_fold_%j.log

# Usage: sbatch scripts/train_usleep_fold.slurm <FOLD>
set -e
FOLD=${1:?Usage: sbatch train_usleep_fold.slurm <FOLD>}

cd /users/hamehdi/projects/sleepfm-mesa
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate usleep_env

PROJECTS_ROOT=/scratch/project_2019517/usleep_mesa/projects
PROJECT_DIR=${PROJECTS_ROOT}/fold_${FOLD}

if [ ! -d "${PROJECT_DIR}" ]; then
    echo "===== Initializing U-Sleep project for fold ${FOLD} ====="
    ut init --name fold_${FOLD} --root "${PROJECTS_ROOT}" --model usleep
fi

echo "===== Preparing fold ${FOLD} view tree + dataset config ====="
python scripts/usleep_prepare_fold.py --fold ${FOLD} --project_dir "${PROJECT_DIR}"

cd "${PROJECT_DIR}"

if ls model/@epoch_*.h5 >/dev/null 2>&1; then
    echo "===== Resuming training for fold ${FOLD} ====="
    ut train --num_gpus 1 --datasets mesa --train_queue_type limitation --continue_training
else
    echo "===== Starting training for fold ${FOLD} ====="
    ut train --num_gpus 1 --datasets mesa --train_queue_type limitation --overwrite
fi

echo "===== Predicting on test split for fold ${FOLD} ====="
ut predict --num_gpus 1 --datasets mesa --data_split test_data \
    --out_dir predictions/test --majority --save_true --overwrite

echo "===== Evaluating fold ${FOLD} ====="
cd /users/hamehdi/projects/sleepfm-mesa
python scripts/evaluate_usleep_fold.py --fold ${FOLD} --project_dir "${PROJECT_DIR}"

echo "========== DONE fold ${FOLD} =========="
