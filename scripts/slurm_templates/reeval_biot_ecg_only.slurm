#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=reeval_biot_ecg_only
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/reeval_biot_ecg_only_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

CKPT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/biot__ECG_ONLY__finetuned__fold5_v1__2026-08-02_1314"

for FOLD in fold_0 fold_1 fold_3; do
    echo "===== Re-evaluating BIOT ECG_ONLY ${FOLD} (results-collision fix) ====="
    python scripts/evaluate_biot.py \
        --modality ECG_ONLY \
        --fold_key "${FOLD}" \
        --checkpoint_dir "$CKPT_DIR" \
        --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full
    EVALEXIT=$?
    if [ $EVALEXIT -ne 0 ]; then
        echo "FATAL: evaluate_biot.py exited with code $EVALEXIT for ${FOLD}"
        exit $EVALEXIT
    fi
done
echo "REEVAL_DONE"
