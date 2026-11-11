#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smoketest_build_token_cache
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/smoketest_build_token_cache_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module load python-data/3.12-31.03

echo "SMOKETEST_START: $(date +%s)"
python scripts/build_token_cache.py \
    --cache_dir /scratch/project_2019517/sleepfm-data/spectral_cache_smoketest \
    --report_name token_cache_report_smoketest.txt
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: build_token_cache.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "SMOKETEST_END: $(date +%s)"
