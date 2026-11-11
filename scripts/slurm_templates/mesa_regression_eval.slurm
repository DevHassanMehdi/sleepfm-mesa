#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_regress
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/mesa_regress_%j.log

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

OUT_DIR=${1:?Usage: sbatch mesa_regression_eval.slurm <OUT_DIR>}

echo "START: $(date +%s)"
python3 sleepfm/pipeline/evaluate_sleep_staging.py \
    --output_path "$OUT_DIR" \
    --split test \
    --fold 0
EXIT=$?
echo "END: $(date +%s) exit=$EXIT"
exit $EXIT
