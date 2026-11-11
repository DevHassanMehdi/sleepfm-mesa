#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_biot_wave1
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moody_biot_wave1_%x_%j.log

# BIOT Wave 1: fold 0, one of the 4 conditions, patience 20. Resumable
# across 18h allocations without a code change to finetune_biot.py: a
# pointer file (owner-only, under the Moody retrain area) records the
# checkpoint_dir the FIRST submission creates (parsed from its own
# ">>> OUTPUT PATH:" print, which appears within seconds of startup,
# well before a timeout could ever hit), and a resubmission reads it
# back and passes --checkpoint_dir so finetune_biot.py's own latest.pth
# resume logic takes over from there.
#
# Usage: sbatch --job-name=moody_biot_wave1_<modality> moody_biot_wave1.slurm <MODALITY>

MODALITY=${1:?Usage: sbatch moody_biot_wave1.slurm <MODALITY>}
MOODY_ROOT=/scratch/project_2019517/moody2026_pilot/retrain
POINTER_FILE="${MOODY_ROOT}/.biot_wave1_fold0_${MODALITY}_checkpoint_dir.txt"
CAPTURE_LOG="${MOODY_ROOT}/.biot_wave1_fold0_${MODALITY}_capture_${SLURM_JOB_ID}.log"

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "WAVE1_START: $(date +%s) modality=$MODALITY"

RESUME_ARGS=""
if [ -f "$POINTER_FILE" ]; then
    CKPT_DIR=$(cat "$POINTER_FILE")
    echo ">>> Resuming from pointer file: $CKPT_DIR"
    RESUME_ARGS="--checkpoint_dir $CKPT_DIR"
fi

python scripts/finetune_biot.py \
    --modality "$MODALITY" \
    --fold_key fold_0 \
    --patience 20 \
    --split_id moody_i0006_fold5_v1 \
    --split_path /scratch/project_2019517/moody2026_pilot/splits/dataset_split_moody_i0006_fold5_v1.json \
    --hdf5_dir /scratch/project_2019517/moody2026_pilot/hdf5_included \
    --labels_dir /scratch/project_2019517/moody2026_pilot/labels_included \
    --root "$MOODY_ROOT" \
    $RESUME_ARGS \
    2>&1 | tee "$CAPTURE_LOG" | awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }' &
PYPID=$!

if [ ! -f "$POINTER_FILE" ]; then
    (
        for i in $(seq 1 90); do
            if grep -q ">>> OUTPUT PATH:" "$CAPTURE_LOG" 2>/dev/null; then
                grep ">>> OUTPUT PATH:" "$CAPTURE_LOG" | head -1 | sed 's/.*OUTPUT PATH: //' | sed 's|/fold_0$||' > "$POINTER_FILE"
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

echo "WAVE1_END: $(date +%s) exit=$PYEXIT modality=$MODALITY"
exit $PYEXIT
