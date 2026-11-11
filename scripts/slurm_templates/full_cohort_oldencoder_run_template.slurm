#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=full_cohort_oldencoder_run
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/full_cohort_oldencoder_run_%j.log

# Experiment A: pretrain_method hardcoded to "fromscratch_oldencoder" so
# results/checkpoints never collide with the full-cohort-encoder results.
#
# Positional args:
#   $1 = config_path, $2 = split_path, $3 = split_id, $4 = fold,
#   $5 = canonical_modality, $6 = timestamp
CONFIG_PATH=$1
SPLIT_PATH=$2
SPLIT_ID=$3
FOLD=$4
CANONICAL_MODALITY=$5
TIMESTAMP=$6

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

CHECKPOINT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__${CANONICAL_MODALITY}__fromscratch_oldencoder__${SPLIT_ID}__${TIMESTAMP}"

echo "RUN_START: $(date +%s)"
echo ">>> modality=$CANONICAL_MODALITY split_id=$SPLIT_ID fold=$FOLD pretrain=fromscratch_oldencoder"
echo ">>> Expected checkpoint_dir: $CHECKPOINT_DIR"

echo "=== STEP 1: finetune_sleep_staging.py ==="
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path "$CONFIG_PATH" \
    --split_path "$SPLIT_PATH" \
    --fold "$FOLD" \
    --pretrain_approach fromscratch_oldencoder \
    --split_id "$SPLIT_ID" \
    --timestamp "$TIMESTAMP"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "FATAL: expected checkpoint_dir does not exist: $CHECKPOINT_DIR"
    exit 1
fi

echo "=== STEP 2: evaluate_sleep_staging.py (val) ==="
python sleepfm/pipeline/evaluate_sleep_staging.py \
    --config_path "$CONFIG_PATH" \
    --output_path "$CHECKPOINT_DIR" \
    --split validation \
    --fold "$FOLD"

echo "=== STEP 3: evaluate_sleep_staging.py (test) ==="
python sleepfm/pipeline/evaluate_sleep_staging.py \
    --config_path "$CONFIG_PATH" \
    --output_path "$CHECKPOINT_DIR" \
    --split test \
    --fold "$FOLD"

echo "=== STEP 4: compute_metrics.py ==="
python scripts/compute_metrics.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --split test \
    --fold "$FOLD"

echo "RUN_END: $(date +%s)"
echo ">>> checkpoint_dir: $CHECKPOINT_DIR"
