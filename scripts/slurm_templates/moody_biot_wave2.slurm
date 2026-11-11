#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_biot_wave2
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/moody_biot_wave2_%x_%j.log

# BIOT Wave 2: one (modality, fold) pair, patience 20. Same resume
# mechanism as Wave 1 (pointer file keyed by modality+fold this time,
# since Wave 2 has multiple folds per modality). evaluate_biot.py runs
# immediately after training, same Moody-only output rules (--root).
#
# Usage: sbatch --job-name=moody_biot_wave2_<modality>_fold<N> moody_biot_wave2.slurm <MODALITY> <FOLD>

MODALITY=${1:?Usage: sbatch moody_biot_wave2.slurm <MODALITY> <FOLD>}
FOLD=${2:?Usage: sbatch moody_biot_wave2.slurm <MODALITY> <FOLD>}
MOODY_ROOT=/scratch/project_2019517/moody2026_pilot/retrain
SPLIT_PATH=/scratch/project_2019517/moody2026_pilot/splits/dataset_split_moody_i0006_fold5_v1.json
HDF5_DIR=/scratch/project_2019517/moody2026_pilot/hdf5_included
LABELS_DIR=/scratch/project_2019517/moody2026_pilot/labels_included
POINTER_FILE="${MOODY_ROOT}/.biot_wave2_${MODALITY}_fold${FOLD}_checkpoint_dir.txt"
CAPTURE_LOG="${MOODY_ROOT}/.biot_wave2_${MODALITY}_fold${FOLD}_capture_${SLURM_JOB_ID}.log"

umask 077
cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "WAVE2_START: $(date +%s) modality=$MODALITY fold=$FOLD"

RESUME_ARGS=""
if [ -f "$POINTER_FILE" ]; then
    CKPT_DIR=$(cat "$POINTER_FILE")
    echo ">>> Resuming from pointer file: $CKPT_DIR"
    RESUME_ARGS="--checkpoint_dir $CKPT_DIR"
fi

python scripts/finetune_biot.py \
    --modality "$MODALITY" \
    --fold_key "fold_${FOLD}" \
    --patience 20 \
    --split_id moody_i0006_fold5_v1 \
    --split_path "$SPLIT_PATH" \
    --hdf5_dir "$HDF5_DIR" \
    --labels_dir "$LABELS_DIR" \
    --root "$MOODY_ROOT" \
    $RESUME_ARGS \
    2>&1 | tee "$CAPTURE_LOG" | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }' &
PYPID=$!

if [ ! -f "$POINTER_FILE" ]; then
    (
        for i in $(seq 1 90); do
            if grep -q ">>> OUTPUT PATH:" "$CAPTURE_LOG" 2>/dev/null; then
                grep ">>> OUTPUT PATH:" "$CAPTURE_LOG" | head -1 | sed 's/.*OUTPUT PATH: //' | sed 's|/fold_'"${FOLD}"'$||' > "$POINTER_FILE"
                chmod 600 "$POINTER_FILE"
                break
            fi
            sleep 2
        done
    ) &
fi

wait $PYPID
PYEXIT=$?
chmod 600 "$CAPTURE_LOG" 2>/dev/null
rm -f "$CAPTURE_LOG"

echo "WAVE2_TRAIN_END: $(date +%s) exit=$PYEXIT modality=$MODALITY fold=$FOLD"

if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: finetune_biot.py exited with code $PYEXIT, skipping evaluation"
    exit $PYEXIT
fi

CKPT_DIR=$(cat "$POINTER_FILE")
echo "=== evaluate_biot.py (test) ==="
python scripts/evaluate_biot.py \
    --modality "$MODALITY" \
    --fold_key "fold_${FOLD}" \
    --checkpoint_dir "$CKPT_DIR" \
    --hdf5_dir "$HDF5_DIR" \
    --labels_dir "$LABELS_DIR" \
    --split_path "$SPLIT_PATH" \
    --root "$MOODY_ROOT"
EVALEXIT=$?

echo "WAVE2_END: $(date +%s) exit=$EVALEXIT modality=$MODALITY fold=$FOLD"
exit $EVALEXIT
