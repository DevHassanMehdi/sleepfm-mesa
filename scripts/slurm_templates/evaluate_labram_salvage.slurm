#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=evaluate_labram_salvage
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/evaluate_labram_salvage_%x_%j.log

# Usage: sbatch evaluate_labram_salvage.slurm <MODALITY> <FOLD_KEY> <CHECKPOINT_DIR>
set -e
MODALITY=${1:?Usage: sbatch evaluate_labram_salvage.slurm <MODALITY> <FOLD_KEY> <CHECKPOINT_DIR>}
FOLD_KEY=${2:?Usage: sbatch evaluate_labram_salvage.slurm <MODALITY> <FOLD_KEY> <CHECKPOINT_DIR>}
CHECKPOINT_DIR=${3:?Usage: sbatch evaluate_labram_salvage.slurm <MODALITY> <FOLD_KEY> <CHECKPOINT_DIR>}

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

echo "SALVAGE_EVAL_START: $(date +%s)  MODALITY=${MODALITY}  FOLD_KEY=${FOLD_KEY}  CHECKPOINT_DIR=${CHECKPOINT_DIR}"
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
echo "SALVAGE_EVAL_END: $(date +%s)"
