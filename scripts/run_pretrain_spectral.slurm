#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_spectral_pretrain
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/pretrain_spectral_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

CKPT_DIR=/scratch/project_2019517/sleepfm-data/checkpoints/spectral/SetTransformer/spectral_128_patch_size_640

if [ -f "${CKPT_DIR}/checkpoint.pt" ]; then
    echo "Resuming from ${CKPT_DIR}"
    python sleepfm/pipeline/pretrain_spectral.py \
        --config_path sleepfm/configs/config_pretrain_spectral.yaml \
        --checkpoint_path ${CKPT_DIR}
else
    echo "Starting fresh"
    python sleepfm/pipeline/pretrain_spectral.py \
        --config_path sleepfm/configs/config_pretrain_spectral.yaml
fi
