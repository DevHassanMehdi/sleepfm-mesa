#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=1-12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=finetune_sleep_staging_fullcohort
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/finetune_sleep_staging_fullcohort_%x_%j.log

# Usage: sbatch finetune_sleep_staging_fullcohort.slurm <CONFIG_PATH> <PRETRAIN_APPROACH> <FOLD>
set -e
CONFIG_PATH=${1:?Usage: sbatch finetune_sleep_staging_fullcohort.slurm <CONFIG_PATH> <PRETRAIN_APPROACH> <FOLD>}
PRETRAIN_APPROACH=${2:?Usage: sbatch finetune_sleep_staging_fullcohort.slurm <CONFIG_PATH> <PRETRAIN_APPROACH> <FOLD>}
FOLD=${3:?Usage: sbatch finetune_sleep_staging_fullcohort.slurm <CONFIG_PATH> <PRETRAIN_APPROACH> <FOLD>}

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

echo "FULLCOHORT_START: $(date +%s)  CONFIG=${CONFIG_PATH}  PRETRAIN_APPROACH=${PRETRAIN_APPROACH}  FOLD=${FOLD}"
FT_LOG=$(mktemp)
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path "${CONFIG_PATH}" \
    --split_id fold5_v1 \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --fold "${FOLD}" \
    --pretrain_approach "${PRETRAIN_APPROACH}" | tee "$FT_LOG"
PYEXIT=${PIPESTATUS[0]}
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: finetune_sleep_staging.py exited with code $PYEXIT"
    exit $PYEXIT
fi

FOLD_DIR=$(grep "^>>> OUTPUT PATH:" "$FT_LOG" | sed 's/^>>> OUTPUT PATH: //' | tr -d ' ')
CKPT_DIR=$(dirname "$FOLD_DIR")
echo "Parsed fold dir: $FOLD_DIR"
echo "Parsed experiment (checkpoint) dir: $CKPT_DIR"
rm -f "$FT_LOG"

echo "===== Evaluating (fold ${FOLD}) ====="
python sleepfm/pipeline/evaluate_sleep_staging.py \
    --output_path "$CKPT_DIR" \
    --fold "${FOLD}" \
    --split test

echo "===== Computing metrics (fold ${FOLD}) ====="
python scripts/compute_metrics.py \
    --checkpoint_dir "$CKPT_DIR" \
    --fold "${FOLD}" \
    --split test

echo "FULLCOHORT_END: $(date +%s)"
