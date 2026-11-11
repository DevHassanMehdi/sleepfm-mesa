#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=smoketest_fc
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/smoketest_fullcohort_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

TIMESTAMP=$(date +%Y-%m-%d_%H%M)
CHECKPOINT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__EEG_ONLY__fromscratch__fold5_v1__${TIMESTAMP}"

echo "SMOKETEST_START: $(date +%s)"
echo ">>> Expected checkpoint_dir: $CHECKPOINT_DIR"

echo "=== STEP 1: finetune_sleep_staging.py ==="
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path sleepfm/configs/config_finetune_full_cohort_smoketest.yaml \
    --split_path sleepfm/configs/dataset_split_fold5_v1.json \
    --fold 0 \
    --pretrain_approach fromscratch \
    --split_id fold5_v1 \
    --timestamp "$TIMESTAMP"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "FATAL: expected checkpoint_dir does not exist: $CHECKPOINT_DIR"
    exit 1
fi

echo "=== STEP 2: evaluate_sleep_staging.py (val) ==="
python sleepfm/pipeline/evaluate_sleep_staging.py \
    --config_path sleepfm/configs/config_finetune_full_cohort_smoketest.yaml \
    --output_path "$CHECKPOINT_DIR" \
    --split validation \
    --fold 0

echo "=== STEP 3: evaluate_sleep_staging.py (test) ==="
python sleepfm/pipeline/evaluate_sleep_staging.py \
    --config_path sleepfm/configs/config_finetune_full_cohort_smoketest.yaml \
    --output_path "$CHECKPOINT_DIR" \
    --split test \
    --fold 0

echo "=== STEP 4: compute_metrics.py ==="
python scripts/compute_metrics.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --split test \
    --n_folds 1

echo "SMOKETEST_END: $(date +%s)"
echo ">>> checkpoint_dir: $CHECKPOINT_DIR"
