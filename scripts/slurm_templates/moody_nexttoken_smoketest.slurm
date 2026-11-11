#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_nexttoken_smoketest
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moody_nexttoken_smoketest_%j.log

# 3-epoch Next-Token pretraining smoke test against the real (reseeded)
# Moody token cache. Writes to its own save_path (nexttoken_moody_smoketest),
# never the real production encoder's directory.

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "SMOKETEST_START: $(date +%s)"
python sleepfm/pipeline/pretrain_nexttoken.py \
    --config_path sleepfm/configs/config_pretrain_nexttoken_moody_smoketest.yaml
EXIT=$?
echo "SMOKETEST_END: $(date +%s) exit=$EXIT"
exit $EXIT
