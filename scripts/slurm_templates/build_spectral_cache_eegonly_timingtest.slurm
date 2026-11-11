#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=build_spectral_cache_eegonly_timingtest
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/build_spectral_cache_eegonly_timingtest_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
module load python-data/3.12-31.03

echo "TIMINGTEST_START: $(date +%s)"

python3 scripts/build_spectral_cache.py \
    --config_path sleepfm/configs/config_pretrain_spectral_eegonly_fullcohort_timingtest.yaml \
    --cache_dir /scratch/project_2019517/sleepfm-data/spectral_cache_eegonly_fullcohort_timingtest

echo "TIMINGTEST_END: $(date +%s)"
