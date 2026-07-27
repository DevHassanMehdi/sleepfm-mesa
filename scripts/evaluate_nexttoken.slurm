#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eval_nexttoken
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/eval_nexttoken_%j.log

# Evaluate the most recent combined fine-tuned run for each modality.
# Each modality's latest run_YYYY-MM-DD_HHMM folder is auto-detected.
# evaluate_sleep_staging.py appends /fold_0 internally; pickles land in
#   {run_dir}/fold_0/mesa/test/
# To evaluate a specific run instead of the latest, set RUN_OVERRIDE:
#   RUN_OVERRIDE=/scratch/.../finetuned/combined/EEG_ONLY/run_2026-07-20_1200

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

CKPT=/scratch/project_2019517/sleepfm-data/checkpoints
PRETRAIN=nexttoken

for MODALITY in EEG_ONLY EKG EEG_ONLY_EKG; do
    if [ -n "${RUN_OVERRIDE}" ]; then
        RUN_DIR="${RUN_OVERRIDE}"
    else
        RUN_DIR=$(ls -d ${CKPT}/finetuned/${PRETRAIN}/${MODALITY}/run_* 2>/dev/null | sort | tail -1)
    fi

    if [ -z "${RUN_DIR}" ]; then
        echo "ERROR: No run found in ${CKPT}/finetuned/${PRETRAIN}/${MODALITY}/" >&2
        exit 1
    fi

    echo "===== Evaluating ${MODALITY}: ${RUN_DIR} ====="
    python sleepfm/pipeline/evaluate_sleep_staging.py \
        --output_path ${RUN_DIR} \
        --split test --fold 0

    echo "===== Metrics for ${MODALITY} ====="
    python scripts/compute_metrics.py \
        --checkpoint_dir ${RUN_DIR} \
        --n_folds 1
done

echo "========== DONE =========="
