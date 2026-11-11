#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_finetune_nexttoken
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/moody_finetune_nexttoken_%j.log

# Next-Token Moody fine-tuning at patience 50 -- same template as
# moody_finetune_run_template.slurm (the From-Scratch patience-50 runs),
# with --pretrain_approach nexttoken (matching MESA's own naming
# convention for Next-Token runs) so the checkpoint_dir/run_name can
# never collide with the From-Scratch patience-50 runs (..._fromscratch__...)
# or the From-Scratch patience-20 runs (..._fromscratch_moody_p20__...).
#
# Positional args:
#   $1 = config_path        (per-modality Moody Next-Token config)
#   $2 = split_path          (dataset_split_moody_i0006_fold5_v1.json)
#   $3 = split_id            ("moody_i0006_fold5_v1")
#   $4 = fold                (integer 0-4)
#   $5 = canonical_modality  (EEG_ONLY | ECG_ONLY | EEG_ECG | EEG3_ONLY)
#   $6 = timestamp           (YYYY-MM-DD_HHMMSS, precomputed by the
#                             submitting script to avoid collisions)
CONFIG_PATH=$1
SPLIT_PATH=$2
SPLIT_ID=$3
FOLD=$4
CANONICAL_MODALITY=$5
TIMESTAMP=$6
PRETRAIN_APPROACH=nexttoken

MOODY_ROOT=/scratch/project_2019517/moody2026_pilot/retrain

umask 077
cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

CHECKPOINT_DIR="${MOODY_ROOT}/checkpoints/full_cohort/sleepfm__${CANONICAL_MODALITY}__${PRETRAIN_APPROACH}__${SPLIT_ID}__${TIMESTAMP}"

echo "RUN_START: $(date +%s)"
echo ">>> modality=$CANONICAL_MODALITY split_id=$SPLIT_ID fold=$FOLD root=$MOODY_ROOT pretrain_approach=$PRETRAIN_APPROACH"
echo ">>> Expected checkpoint_dir: $CHECKPOINT_DIR"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "FATAL: checkpoint_dir already exists -- refusing to risk colliding with a prior run: $CHECKPOINT_DIR"
    exit 1
fi

echo "=== STEP 1: finetune_sleep_staging.py ==="
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path "$CONFIG_PATH" \
    --split_path "$SPLIT_PATH" \
    --fold "$FOLD" \
    --pretrain_approach "$PRETRAIN_APPROACH" \
    --split_id "$SPLIT_ID" \
    --timestamp "$TIMESTAMP" \
    --patience 50 \
    --root "$MOODY_ROOT" \
    --split_tmp_dir "${MOODY_ROOT}/tmp_splits"

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
    --fold "$FOLD" \
    --root "$MOODY_ROOT"

echo "RUN_END: $(date +%s)"
echo ">>> checkpoint_dir: $CHECKPOINT_DIR"
