#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=p20match
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/p20match_%x_%j.log

# Matched-patience-20 runs for MESA From-Scratch, MESA Next-Token, and
# Moody Next-Token (the three new batches requested). Shared template:
# ROOT="NONE" writes to the default shared MESA tree
# (checkpoints/full_cohort/, outside moody2026_pilot -- "MESA output in
# its own area"); any other ROOT value is passed through to --root
# (Moody area, owner-only, umask 077 below -- "Moody output in its own
# area"). pretrain_approach is caller-supplied and must be a NEW name
# distinct from every existing pretrain_approach for that
# model/modality/split_id, so the checkpoint_dir can never collide with
# the 90-run benchmark, the single-channel comparison reruns, or any
# prior Moody batch -- enforced by the explicit pre-existence check
# below, not just by convention.
#
# Positional args:
#   $1 = config_path
#   $2 = split_path
#   $3 = split_id
#   $4 = fold
#   $5 = canonical_modality
#   $6 = timestamp
#   $7 = pretrain_approach   (e.g. fromscratch_p20, nexttoken_p20, nexttoken_moody_p20)
#   $8 = root                ("NONE" for MESA, or a Moody root path)
CONFIG_PATH=$1
SPLIT_PATH=$2
SPLIT_ID=$3
FOLD=$4
CANONICAL_MODALITY=$5
TIMESTAMP=$6
PRETRAIN_APPROACH=$7
ROOT_ARG=$8

umask 077
cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

if [ "$ROOT_ARG" == "NONE" ]; then
    BASE_CKPT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort"
    ROOT_FLAG=""
    EVAL_ROOT_FLAG=""
else
    BASE_CKPT_DIR="${ROOT_ARG}/checkpoints/full_cohort"
    ROOT_FLAG="--root ${ROOT_ARG} --split_tmp_dir ${ROOT_ARG}/tmp_splits"
    EVAL_ROOT_FLAG="--root ${ROOT_ARG}"
fi

CHECKPOINT_DIR="${BASE_CKPT_DIR}/sleepfm__${CANONICAL_MODALITY}__${PRETRAIN_APPROACH}__${SPLIT_ID}__${TIMESTAMP}"

echo "RUN_START: $(date +%s)"
echo ">>> modality=$CANONICAL_MODALITY split_id=$SPLIT_ID fold=$FOLD pretrain_approach=$PRETRAIN_APPROACH root=$ROOT_ARG"
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
    --patience 20 \
    $ROOT_FLAG

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
    $EVAL_ROOT_FLAG

echo "RUN_END: $(date +%s)"
echo ">>> checkpoint_dir: $CHECKPOINT_DIR"
