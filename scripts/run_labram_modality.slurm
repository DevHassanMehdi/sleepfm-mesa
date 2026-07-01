#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=labram_modality
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/labram_modality_%j.log

# Usage: sbatch run_labram_modality.slurm <MODALITY>
# MODALITY: EEG_ONLY only -- LaBraM is an EEG-only foundation model, and
# MESA's EOG channels have no defensible mapping into its 10-20-indexed
# vocabulary (see labram_dataset.py docstring).
set -e
MODALITY=${1:?Usage: sbatch run_labram_modality.slurm <MODALITY>}

cd /users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

echo "===== Fine-tuning LaBraM (${MODALITY}) ====="
python scripts/finetune_labram.py --modality ${MODALITY}

echo "===== Evaluating LaBraM (${MODALITY}) ====="
python scripts/evaluate_labram.py --modality ${MODALITY}

echo "========== DONE (${MODALITY}) =========="
