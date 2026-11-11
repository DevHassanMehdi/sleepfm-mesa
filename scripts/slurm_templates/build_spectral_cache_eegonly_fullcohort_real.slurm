#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=build_spectral_cache_eegonly_fullcohort
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/build_spectral_cache_eegonly_fullcohort_real_%j.log

cd /users/hamehd/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
module load python-data/3.12-31.03

echo "REALBUILD_START: $(date +%s)"

python3 scripts/build_spectral_cache.py \
    --config_path sleepfm/configs/config_pretrain_spectral_eegonly_fullcohort.yaml \
    --cache_dir /scratch/project_2019517/sleepfm-data/spectral_cache_eegonly_fullcohort

echo "REALBUILD_END: $(date +%s)"
