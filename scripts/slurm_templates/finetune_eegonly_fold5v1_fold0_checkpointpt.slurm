#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ft_eegonly_fold5v1_f0_ckptpt
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/finetune_eegonly_fold5v1_fold0_checkpointpt_%j.log

# checkpoint.pt vs best.pt A/B test: identical config/fold/hyperparameters to
# the completed best.pt-based run (sleepfm__EEG_ONLY__fromscratch__fold5_v1__
# 2026-07-29_150556), only model_path differs (points at the checkpoint.pt-
# derived embeddings cache via a symlinked wrapper dir).
#
# NOTE: finetune_sleep_staging.py derives the modality string internally from
# config["channel_like"] ("_".join -> canonical_modality()), NOT from a CLI
# arg -- so this run's checkpoint_dir will use modality="EEG_ONLY", identical
# to the original best.pt run. Distinctness comes solely from TIMESTAMP
# (guaranteed unique), exactly as the rest of this pipeline already relies on
# for concurrent/repeated same-modality/split runs.

CONFIG_PATH="sleepfm/configs/config_finetune_full_cohort_EEG_ONLY_checkpointpt.yaml"
SPLIT_PATH="sleepfm/configs/dataset_split_fold5_v1.json"
SPLIT_ID="fold5_v1"
FOLD=0
CANONICAL_MODALITY="EEG_ONLY"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

CHECKPOINT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__${CANONICAL_MODALITY}__fromscratch__${SPLIT_ID}__${TIMESTAMP}"

echo "RUN_START: $(date +%s)"
echo ">>> modality=$CANONICAL_MODALITY split_id=$SPLIT_ID fold=$FOLD"
echo ">>> Expected checkpoint_dir: $CHECKPOINT_DIR"

echo "=== STEP 1: finetune_sleep_staging.py ==="
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path "$CONFIG_PATH" \
    --split_path "$SPLIT_PATH" \
    --fold "$FOLD" \
    --pretrain_approach fromscratch \
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
