#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=moody_build_token_cache
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/moody_build_token_cache_%j.log

# Fits MiniBatchKMeans (n_clusters=512, same as MESA) on the Moody spectral
# cache's pretrain split, assigns tokens to both splits, writes
# token_codebook.npy / token_pairs_{pretrain,validation}.npy /
# token_assignments_{pretrain,validation}.npy into the same Moody-owned
# cache_dir. Submitted with --dependency=afterok on the spectral cache job.

umask 077
mkdir -p /scratch/project_2019517/moody2026_pilot/retrain/diagnostics
chmod 700 /scratch/project_2019517/moody2026_pilot/retrain/diagnostics
module use /appl/modulefiles/manual/aida/x86_64
module load python-pytorch/2.10
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "TOKEN_CACHE_START: $(date +%s)"

# cd into a Moody-owned dir (not the git repo) so build_token_cache.py's
# hardcoded relative "diagnostics/" report dir lands outside git, matching
# every other Moody artifact in this project.
cd /scratch/project_2019517/moody2026_pilot/retrain
python /users/hamehd/projects/sleepfm-mesa/scripts/build_token_cache.py \
    --cache_dir /scratch/project_2019517/moody2026_pilot/retrain/nexttoken_cache \
    --report_name token_cache_report_moody.txt
EXIT=$?

chmod -R 600 /scratch/project_2019517/moody2026_pilot/retrain/nexttoken_cache/* 2>/dev/null
chmod 600 /scratch/project_2019517/moody2026_pilot/retrain/diagnostics/token_cache_report_moody.txt 2>/dev/null
echo "TOKEN_CACHE_END: $(date +%s) exit=$EXIT"
exit $EXIT
