#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=build_token_cache_fullcohort
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/build_token_cache_fullcohort_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module load python-data/3.12-31.03

echo "FULLCOHORT_START: $(date +%s)"
python scripts/build_token_cache.py \
    --cache_dir /scratch/project_2019517/sleepfm-data/spectral_cache_eegonly_fullcohort \
    --report_name token_cache_report_fullcohort.txt
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: build_token_cache.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "FULLCOHORT_END: $(date +%s)"
