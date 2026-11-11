#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=labram_resume
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/labram_resume_%x_%j.log

# Usage: sbatch --time=<H:MM:SS> [--dependency=afterany:<jid>] finetune_labram_resume.slurm <MODALITY> <FOLD_KEY> <CHECKPOINT_DIR> [--eval]
# CHECKPOINT_DIR is the PARENT run directory (no /fold_N suffix).
# Pass a 4th arg "eval" to also run evaluate_labram.py after training exits
# (only pass this on the LAST job in a chain, so eval only runs once).
set -e
MODALITY=${1:?Usage: sbatch finetune_labram_resume.slurm <MODALITY> <FOLD_KEY> <CHECKPOINT_DIR> [eval]}
FOLD_KEY=${2:?Usage: sbatch finetune_labram_resume.slurm <MODALITY> <FOLD_KEY> <CHECKPOINT_DIR> [eval]}
CHECKPOINT_DIR=${3:?Usage: sbatch finetune_labram_resume.slurm <MODALITY> <FOLD_KEY> <CHECKPOINT_DIR> [eval]}
RUN_EVAL=${4:-noeval}

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

echo "RESUME_START: $(date +%s)  MODALITY=${MODALITY}  FOLD_KEY=${FOLD_KEY}  CHECKPOINT_DIR=${CHECKPOINT_DIR}  RUN_EVAL=${RUN_EVAL}"
python scripts/finetune_labram.py \
    --modality "${MODALITY}" \
    --fold_key "${FOLD_KEY}" \
    --split_id fold5_v1 \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --checkpoint_dir "${CHECKPOINT_DIR}"
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: finetune_labram.py exited with code $PYEXIT"
    exit $PYEXIT
fi

if [ "$RUN_EVAL" == "eval" ]; then
    echo "===== Evaluating LaBraM (${MODALITY}, ${FOLD_KEY}) after resume ====="
    python scripts/evaluate_labram.py \
        --modality "${MODALITY}" \
        --fold_key "${FOLD_KEY}" \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full
    EVALEXIT=$?
    if [ $EVALEXIT -ne 0 ]; then
        echo "FATAL: evaluate_labram.py exited with code $EVALEXIT"
        exit $EVALEXIT
    fi
fi
echo "RESUME_END: $(date +%s)"
