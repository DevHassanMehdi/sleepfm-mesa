#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_nextwindow_pretrain
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/pretrain_nextwindow_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

CONFIG=sleepfm/configs/config_pretrain_nextwindow.yaml
CHANNELS=sleepfm/configs/channel_groups.json
CKPT_DIR=/scratch/project_2019517/sleepfm-data/checkpoints/nextwindow/SetTransformer/nextwindow_128_patch_size_640

if [ -f "${CKPT_DIR}/checkpoint.pt" ]; then
    echo "Resuming from ${CKPT_DIR}/checkpoint.pt"
    python sleepfm/pipeline/pretrain_nextwindow.py \
        --config_path ${CONFIG} \
        --channel_groups_path ${CHANNELS} \
        --checkpoint_path ${CKPT_DIR}
else
    echo "Starting fresh next-window pretraining"
    python sleepfm/pipeline/pretrain_nextwindow.py \
        --config_path ${CONFIG} \
        --channel_groups_path ${CHANNELS}
fi
