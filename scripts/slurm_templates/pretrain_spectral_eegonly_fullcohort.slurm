#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=1-12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pretrain_spectral_eegonly_fullcohort
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/pretrain_spectral_eegonly_fullcohort_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "FULLCOHORT_START: $(date +%s)"
CKPT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/encoders/spectral_eegonly_fullcohort/SetTransformer/spectral_128_patch_size_640"
if [ -f "${CKPT_DIR}/checkpoint.pt" ]; then
    echo "Resuming from ${CKPT_DIR}"
    python sleepfm/pipeline/pretrain_spectral.py \
        --config_path sleepfm/configs/config_pretrain_spectral_eegonly_fullcohort.yaml \
        --channel_groups_path sleepfm/configs/channel_groups.json \
        --checkpoint_path "${CKPT_DIR}"
else
    echo "Starting fresh"
    python sleepfm/pipeline/pretrain_spectral.py \
        --config_path sleepfm/configs/config_pretrain_spectral_eegonly_fullcohort.yaml \
        --channel_groups_path sleepfm/configs/channel_groups.json
fi
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: pretrain_spectral.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "FULLCOHORT_END: $(date +%s)"
