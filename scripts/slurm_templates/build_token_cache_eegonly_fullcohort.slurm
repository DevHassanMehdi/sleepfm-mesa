#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=build_token_cache_eegonly_fullcohort
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/build_token_cache_eegonly_fullcohort_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module load python-data/3.12-31.03

echo "TOKENCACHE_START: $(date +%s)"

python3 scripts/build_token_cache.py \
    --cache_dir /scratch/project_2019517/sleepfm-data/spectral_cache_eegonly_fullcohort \
    --report_name token_cache_report_eegonly_fullcohort.txt

echo "TOKENCACHE_END: $(date +%s)"
