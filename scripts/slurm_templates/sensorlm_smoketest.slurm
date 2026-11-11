#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sensorlm_smoketest
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/sensorlm_smoketest_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

echo "SMOKETEST_START: $(date +%s)"
FT_LOG=$(mktemp)
python scripts/finetune_sensorlm.py \
    --modality EEG_ONLY \
    --fold_key fold_0 \
    --split_id fold5_v1 \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --epochs 1 \
    --patience 1 | tee "$FT_LOG"
PYEXIT=${PIPESTATUS[0]}
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: finetune_sensorlm.py exited with code $PYEXIT"
    exit $PYEXIT
fi

FOLD_DIR=$(grep "^>>> OUTPUT PATH:" "$FT_LOG" | sed 's/^>>> OUTPUT PATH: //')
CKPT_DIR=$(dirname "$FOLD_DIR")
echo "Parsed fold dir: $FOLD_DIR"
echo "Parsed experiment (checkpoint) dir: $CKPT_DIR"
rm -f "$FT_LOG"

echo "===== Evaluating SensorLM (EEG_ONLY) ====="
python scripts/evaluate_sensorlm.py \
    --modality EEG_ONLY \
    --fold_key fold_0 \
    --checkpoint_dir "$CKPT_DIR" \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full
EVALEXIT=$?
if [ $EVALEXIT -ne 0 ]; then
    echo "FATAL: evaluate_sensorlm.py exited with code $EVALEXIT"
    exit $EVALEXIT
fi
echo "SMOKETEST_END: $(date +%s)"
