#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smoketest_pretrain_nexttoken
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/smoketest_pretrain_nexttoken_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "SMOKETEST_START: $(date +%s)"
python sleepfm/pipeline/pretrain_nexttoken.py \
    --config_path sleepfm/configs/config_pretrain_nexttoken_eegonly_smoketest.yaml
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: pretrain_nexttoken.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "SMOKETEST_END: $(date +%s)"
