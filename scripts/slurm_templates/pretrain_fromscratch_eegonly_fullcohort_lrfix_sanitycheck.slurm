#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fs_lrfix_sanitycheck
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/pretrain_fromscratch_lrfix_sanitycheck_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "SANITYCHECK_START: $(date +%s)"
python sleepfm/pipeline/pretrain.py \
    --config_path /scratch/project_2019517/claude_arm_build/config_pretrain_fromscratch_eegonly_fullcohort_lrfix_sanitycheck.yaml \
    --channel_groups_path sleepfm/configs/channel_groups.json
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: pretrain.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "SANITYCHECK_END: $(date +%s)"
