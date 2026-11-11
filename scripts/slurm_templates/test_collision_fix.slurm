#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test_collision
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/test_collision_%j.log

# Deliberately uses the smoketest config (epochs=3) purely to stress-test
# the split_id/fold collision fix in finetune_sleep_staging.py -- two jobs
# with SAME fold number, DIFFERENT split_id, submitted simultaneously.
# Not a real training run; deleted after validation.
SPLIT_ID=$1
SPLIT_PATH=$2

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)_$$
echo "COLLISION_TEST_START split_id=$SPLIT_ID fold=0 pid=$$"

python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path sleepfm/configs/config_finetune_full_cohort_smoketest.yaml \
    --split_path "$SPLIT_PATH" \
    --fold 0 \
    --pretrain_approach fromscratch \
    --split_id "collision_test_${SPLIT_ID}" \
    --timestamp "$TIMESTAMP" 2>&1 | grep -E "Data: fold=|OUTPUT PATH"

echo "COLLISION_TEST_END split_id=$SPLIT_ID"
