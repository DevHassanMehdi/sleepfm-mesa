#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_eeg3_p50ref
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/mesa_eeg3_p50ref_%j.log

# MESA EEG3-only reference at patience 50 (From-Scratch), using the
# EXISTING channel-comparison EEG3 embeddings (not deleted, not
# regenerated). pretrain_method name "fromscratch_ch3_p50ref" keeps the
# run_name distinct from both the 90-run benchmark
# (sleepfm__EEG3_ONLY__fromscratch__fold5_v1__*, doesn't exist anyway --
# EEG3_ONLY was never part of the benchmark) and the channel-comparison
# patience-10 runs (sleepfm__EEG3_ONLY__fromscratch__fold5_v1__2026-10-27_153632_x).
#
# Usage: sbatch mesa_eeg3_p50ref.slurm <FOLD> <TIMESTAMP>
FOLD=$1
TIMESTAMP=$2
CONFIG_PATH=sleepfm/configs/config_ft_fromscratch_eegonly_fullcohort_EEG3_ONLY.yaml
SPLIT_PATH=sleepfm/configs/dataset_split_fold5_v1.json
SPLIT_ID=fold5_v1
CHECKPOINT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__EEG3_ONLY__fromscratch_ch3_p50ref__${SPLIT_ID}__${TIMESTAMP}"

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "RUN_START: $(date +%s) fold=$FOLD"
echo ">>> Expected checkpoint_dir: $CHECKPOINT_DIR"

echo "=== STEP 1: finetune_sleep_staging.py ==="
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path "$CONFIG_PATH" \
    --split_path "$SPLIT_PATH" \
    --fold "$FOLD" \
    --pretrain_approach fromscratch_ch3_p50ref \
    --split_id "$SPLIT_ID" \
    --timestamp "$TIMESTAMP" \
    --patience 50

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
