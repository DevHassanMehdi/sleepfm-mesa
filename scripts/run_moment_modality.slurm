#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moment_modality
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moment_modality_%j.log

# Usage: sbatch run_moment_modality.slurm <MODALITY>
# MODALITY one of: EEG_ONLY, ECG_ONLY, EEG_ECG, BAS, BAS_EKG, BAS_EKG_RESP, BAS_EKG_RESP_EMG
#
# Time budget is 48h (vs BIOT's 24h): MOMENT-1-large fine-tunes a 341M-param
# T5 encoder end-to-end (BIOT's encoder is ~3M params), so each epoch is
# substantially slower despite gradient checkpointing. Batch size is also
# kept conservative (8, vs BIOT's 16) to fit full fine-tuning activations +
# AdamW optimizer state in V100 memory -- see smoke test notes in
# finetune_moment.py before increasing it.
set -e
MODALITY=${1:?Usage: sbatch run_moment_modality.slurm <MODALITY>}

cd /users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

echo "===== Fine-tuning MOMENT (${MODALITY}) ====="
python scripts/finetune_moment.py --modality ${MODALITY}

echo "===== Evaluating MOMENT (${MODALITY}) ====="
python scripts/evaluate_moment.py --modality ${MODALITY}

echo "========== DONE (${MODALITY}) =========="
