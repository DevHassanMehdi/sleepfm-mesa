#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smoketest_spectral_cache
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/smoketest_spectral_cache_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
module load python-data/3.12-31.03

echo "SMOKETEST_START: $(date +%s)"
python3 scripts/build_spectral_cache.py \
    --config_path sleepfm/configs/config_spectral_cache_smoketest.yaml \
    --cache_dir /scratch/project_2019517/sleepfm-data/spectral_cache_smoketest
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: build_spectral_cache.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "SMOKETEST_END: $(date +%s)"
