#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=regtest_b_eval
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/regtest_b_eval_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "START: $(date +%s)"
python sleepfm/pipeline/evaluate_sleep_staging.py \
    --channel_groups_path sleepfm/configs/channel_groups.json \
    --output_path /scratch/project_2019517/regression_test_b \
    --split test \
    --fold 0
echo "END: $(date +%s) exit=$?"
