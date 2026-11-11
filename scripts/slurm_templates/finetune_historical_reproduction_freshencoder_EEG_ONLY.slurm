#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ft_historical_repro_freshenc_EEG_ONLY
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/finetune_historical_reproduction_freshencoder_EEG_ONLY_%j.log

# Experiment C: historical reproduction using the FRESH from-random-init
# encoder (not the reused June one). Same verified-identical 300/30/50
# split as Experiment B.

CONFIG_PATH="sleepfm/configs/config_finetune_historical_reproduction_freshencoder_EEG_ONLY.yaml"
SPLIT_PATH="sleepfm/configs/dataset_split_fromscratch_historical_as_fold0.json"
SPLIT_ID="historical_repro_freshenc"
FOLD=0
CANONICAL_MODALITY="EEG_ONLY"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

CHECKPOINT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__${CANONICAL_MODALITY}__fromscratch_fresh_repro__${SPLIT_ID}__${TIMESTAMP}"

echo "RUN_START: $(date +%s)"
echo ">>> Expected checkpoint_dir: $CHECKPOINT_DIR"

echo "=== STEP 1: finetune_sleep_staging.py ==="
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path "$CONFIG_PATH" \
    --split_path "$SPLIT_PATH" \
    --fold "$FOLD" \
    --pretrain_approach fromscratch_fresh_repro \
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
