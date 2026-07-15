#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_build_spectral_cache
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/build_spectral_cache_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

python scripts/build_spectral_cache.py \
    --config_path sleepfm/configs/config_pretrain_spectral.yaml \
    --cache_dir /scratch/project_2019517/sleepfm-data
