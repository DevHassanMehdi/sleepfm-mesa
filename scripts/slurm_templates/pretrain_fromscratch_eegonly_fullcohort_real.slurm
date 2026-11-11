#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=1-06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pretrain_fs_eegonly_fc
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/pretrain_fromscratch_eegonly_fullcohort_real_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

CKPT_DIR=/scratch/project_2019517/sleepfm-data/checkpoints/encoders/fromscratch_eegonly_fullcohort/SetTransformer/leave_one_out_128_patch_size_640
CONFIG=sleepfm/configs/config_pretrain_fromscratch_eegonly_fullcohort.yaml
CHANNELS=sleepfm/configs/channel_groups.json

echo "REALRUN_START: $(date +%s)"

if [ -f "${CKPT_DIR}/checkpoint.pt" ]; then
    echo "Resuming from ${CKPT_DIR}/checkpoint.pt"
    python sleepfm/pipeline/pretrain.py --config_path ${CONFIG} --channel_groups_path ${CHANNELS} --checkpoint_path ${CKPT_DIR}
else
    echo "Starting fresh pretraining"
    python sleepfm/pipeline/pretrain.py --config_path ${CONFIG} --channel_groups_path ${CHANNELS}
fi

echo "REALRUN_END: $(date +%s)"
