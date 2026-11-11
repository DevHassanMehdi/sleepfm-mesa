#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_biot_timing
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/moody_biot_timing_%j.log

# BIOT timing run: 3-channel EEG, fold 0, patience 20, Moody 5-fold
# split. Runs to actual early-stop convergence (no epoch cap beyond
# finetune_biot.py's own --epochs default of 100) so time/epoch and
# epochs-to-stop are measured, not guessed. Every line is timestamped
# (awk, since finetune_biot.py's own prints have no wall-clock time) so
# time/epoch can be computed directly from this log without touching
# finetune_biot.py itself.

MOODY_ROOT=/scratch/project_2019517/moody2026_pilot/retrain

umask 077
cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "TIMING_START: $(date +%s)"

python scripts/finetune_biot.py \
    --modality EEG_ONLY \
    --fold_key fold_0 \
    --patience 20 \
    --split_id moody_i0006_fold5_v1 \
    --split_path /scratch/project_2019517/moody2026_pilot/splits/dataset_split_moody_i0006_fold5_v1.json \
    --hdf5_dir /scratch/project_2019517/moody2026_pilot/hdf5_included \
    --labels_dir /scratch/project_2019517/moody2026_pilot/labels_included \
    --root "$MOODY_ROOT" \
    2>&1 | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
PYEXIT=${PIPESTATUS[0]}

echo "TIMING_END: $(date +%s) exit=$PYEXIT"
exit $PYEXIT
