#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pretrain_nexttoken_eegonly_fc_timingtest
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/pretrain_nexttoken_eegonly_fullcohort_timingtest_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "TIMINGTEST_START: $(date +%s)"

python sleepfm/pipeline/pretrain_nexttoken.py \
    --config_path sleepfm/configs/config_pretrain_nexttoken_eegonly_fullcohort_timingtest.yaml

echo "TIMINGTEST_END: $(date +%s)"
