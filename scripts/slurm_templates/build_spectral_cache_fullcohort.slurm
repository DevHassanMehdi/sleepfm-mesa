#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=build_spectral_cache_fullcohort
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/build_spectral_cache_fullcohort_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
module load python-data/3.12-31.03

echo "FULLCOHORT_START: $(date +%s)"
python scripts/build_spectral_cache.py \
    --config_path sleepfm/configs/config_pretrain_spectral_eegonly_fullcohort.yaml \
    --cache_dir /scratch/project_2019517/sleepfm-data/spectral_cache_eegonly_fullcohort \
    --val_size 100
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: build_spectral_cache.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "FULLCOHORT_END: $(date +%s)"
