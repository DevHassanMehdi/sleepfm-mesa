#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=labram_resume_ECG_ONLY_fold_4
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/labram_resume_ECG_ONLY_fold_4_%j.log

set -e
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

CKPT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/labram__ECG_ONLY__finetuned__fold5_v1__2026-08-02_1315"

echo "RESUME_START: $(date +%s)"
python scripts/finetune_labram.py \
    --modality ECG_ONLY \
    --fold_key fold_4 \
    --split_id fold5_v1 \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --checkpoint_dir "$CKPT_DIR"

echo "===== Evaluating LaBraM (ECG_ONLY, fold_4) ====="
python scripts/evaluate_labram.py \
    --modality ECG_ONLY \
    --fold_key fold_4 \
    --checkpoint_dir "$CKPT_DIR" \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full
echo "RESUME_END: $(date +%s)"
