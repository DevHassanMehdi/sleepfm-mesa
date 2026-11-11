#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fromscratch_eegonly_fullcohort_seed123
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/fromscratch_eegonly_fullcohort_seed123_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "FULLCOHORT_START: $(date +%s)"
python sleepfm/pipeline/pretrain.py \
    --config_path sleepfm/configs/config_pretrain_fromscratch_eegonly_fullcohort_seed123.yaml \
    --channel_groups_path sleepfm/configs/channel_groups.json \
    --seed 123
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: pretrain.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "FULLCOHORT_END: $(date +%s)"
