#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moment_embed
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moment_embed_%j.log

# Usage: sbatch run_moment_embed.slurm <MODALITY>
# MODALITY one of: EEG_ONLY, ECG_ONLY, EEG_ECG, BAS, BAS_EKG, BAS_EKG_RESP, BAS_EKG_RESP_EMG
#
# Generates pre-computed embeddings for all 3 splits (train/val/test).
# Done-markers prevent re-processing already-completed splits; safe to resubmit.
set -e
MODALITY=${1:?Usage: sbatch run_moment_embed.slurm <MODALITY>}
FOLD_KEY=${2:-fold_0}

cd /users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

echo "===== Generating MOMENT embeddings (${MODALITY}) ====="
python scripts/generate_moment_embeddings.py --modality ${MODALITY} --fold_key ${FOLD_KEY}

echo "========== DONE (${MODALITY}) =========="
