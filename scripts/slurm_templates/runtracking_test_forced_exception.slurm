#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=runtracking_test_forced_exception
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/runtracking_test_forced_exception_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "TEST_START: $(date +%s)"
python sleepfm/pipeline/pretrain.py \
    --config_path sleepfm/configs/config_runtracking_test_forced_exception.yaml \
    --channel_groups_path sleepfm/configs/channel_groups.json
PYEXIT=$?
echo "PYEXIT=$PYEXIT (expected nonzero -- this test intentionally forces an exception)"
echo "TEST_END: $(date +%s)"
