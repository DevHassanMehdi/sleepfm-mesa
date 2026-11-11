#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=biot_fullcohort
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/biot_fullcohort_%x_%j.log

# Usage: sbatch biot_fullcohort.slurm <MODALITY> <FOLD_KEY>
set -e
MODALITY=${1:?Usage: sbatch biot_fullcohort.slurm <MODALITY> <FOLD_KEY>}
FOLD_KEY=${2:?Usage: sbatch biot_fullcohort.slurm <MODALITY> <FOLD_KEY>}

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

echo "FULLCOHORT_START: $(date +%s)  MODALITY=${MODALITY}  FOLD_KEY=${FOLD_KEY}"
FT_LOG=$(mktemp)
python scripts/finetune_biot.py \
    --modality "${MODALITY}" \
    --fold_key "${FOLD_KEY}" \
    --split_id fold5_v1 \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full | tee "$FT_LOG"
PYEXIT=${PIPESTATUS[0]}
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: finetune_biot.py exited with code $PYEXIT"
    exit $PYEXIT
fi

FOLD_DIR=$(grep "^>>> OUTPUT PATH:" "$FT_LOG" | sed 's/^>>> OUTPUT PATH: //')
CKPT_DIR=$(dirname "$FOLD_DIR")
echo "Parsed experiment (checkpoint) dir: $CKPT_DIR"
rm -f "$FT_LOG"

echo "===== Evaluating BIOT (${MODALITY}, ${FOLD_KEY}) ====="
python scripts/evaluate_biot.py \
    --modality "${MODALITY}" \
    --fold_key "${FOLD_KEY}" \
    --checkpoint_dir "$CKPT_DIR" \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full
echo "FULLCOHORT_END: $(date +%s)"
