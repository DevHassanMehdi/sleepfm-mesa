#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=smoketest_valfix
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/smoketest_valfix_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

CHECKPOINT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-07-28_1557"

echo "VALFIX_START: $(date +%s)"
python sleepfm/pipeline/evaluate_sleep_staging.py \
    --config_path sleepfm/configs/config_finetune_full_cohort_smoketest.yaml \
    --output_path "$CHECKPOINT_DIR" \
    --split validation \
    --fold 0
echo "VALFIX_END: $(date +%s)"
