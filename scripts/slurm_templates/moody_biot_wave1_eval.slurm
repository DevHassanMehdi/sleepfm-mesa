#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_biot_wave1_eval
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/moody_biot_wave1_eval_%j.log

# Test-set evaluation for the 4 BIOT Wave 1 fold-0 checkpoints, all on
# the Moody test split, writing per-subject outputs only into the
# Moody-rooted results_dir (--root).

MOODY_ROOT=/scratch/project_2019517/moody2026_pilot/retrain
SPLIT_PATH=/scratch/project_2019517/moody2026_pilot/splits/dataset_split_moody_i0006_fold5_v1.json
HDF5_DIR=/scratch/project_2019517/moody2026_pilot/hdf5_included
LABELS_DIR=/scratch/project_2019517/moody2026_pilot/labels_included

umask 077
cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "EVAL_START: $(date +%s)"

for item in "EEG_ONLY:biot__EEG_ONLY__finetuned__moody_i0006_fold5_v1__2026-11-06_1556" \
            "EEG3_ONLY:biot__EEG3_ONLY__finetuned__moody_i0006_fold5_v1__2026-11-06_1556" \
            "ECG_ONLY:biot__ECG_ONLY__finetuned__moody_i0006_fold5_v1__2026-11-06_1556" \
            "EEG_ECG:biot__EEG_ECG__finetuned__moody_i0006_fold5_v1__2026-11-06_1556"; do
    MODALITY="${item%%:*}"
    DIRNAME="${item#*:}"
    CKPT_DIR="${MOODY_ROOT}/checkpoints/full_cohort/${DIRNAME}"
    echo "=== evaluating ${MODALITY} ==="
    python scripts/evaluate_biot.py \
        --modality "$MODALITY" \
        --fold_key fold_0 \
        --checkpoint_dir "$CKPT_DIR" \
        --hdf5_dir "$HDF5_DIR" \
        --labels_dir "$LABELS_DIR" \
        --split_path "$SPLIT_PATH" \
        --root "$MOODY_ROOT"
    EXIT=$?
    echo "=== ${MODALITY} exit=$EXIT ==="
done

echo "EVAL_END: $(date +%s)"
