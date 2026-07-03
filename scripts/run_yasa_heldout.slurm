#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=yasa_heldout
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/yasa_heldout_%j.log

# Usage: sbatch run_yasa_heldout.slurm <MODALITY>
# MODALITY: EEG_ONLY or EEG_ECG (ECG_ONLY is skipped — YASA requires EEG)
#
# YASA is CPU-only; small partition is sufficient.
# Runtime: ~3-5 min per subject × 50 subjects ≈ 2-4h total.
set -e
MODALITY=${1:?Usage: sbatch run_yasa_heldout.slurm <MODALITY>}

cd /users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

echo "===== YASA held-out evaluation (${MODALITY}) ====="
python scripts/run_yasa_heldout.py --modality ${MODALITY}
echo "========== DONE (${MODALITY}) =========="
