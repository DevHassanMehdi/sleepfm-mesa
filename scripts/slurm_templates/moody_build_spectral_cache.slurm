#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_build_spectral_cache
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/moody_build_spectral_cache_%j.log

# Full-scale (1135 pretrain + 100 validation) Moody spectral cache for
# Next-Token pretraining. Same MODALITY_TYPES/MAX_CHANNELS (EEG_ONLY+EKG,
# 4-channel) as build_spectral_cache.py already hardcodes for MESA -- no
# script change needed, only config_path/cache_dir point at Moody.
# Verified first on a 10-record dry run (test10_hdf5, exit 0, sane spectral
# target stats, ~280 windows/sec on this CPU) before this full submission.
# Not resumable mid-split (build_split_cache only skips a split if BOTH its
# .bin and .npz already exist) -- 12h is set with margin over the ~6.7h
# measured-rate projection for the full cohort, so a resubmission should
# only be needed if something else is unexpectedly slow.

umask 077
mkdir -p /scratch/project_2019517/moody2026_pilot/retrain/nexttoken_cache
chmod 700 /scratch/project_2019517/moody2026_pilot/retrain/nexttoken_cache
cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/x86_64
module load python-pytorch/2.10
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE

echo "SPECTRAL_CACHE_START: $(date +%s)"

python scripts/build_spectral_cache.py \
    --config_path sleepfm/configs/config_pretrain_fromscratch_eegonly_moody.yaml \
    --cache_dir /scratch/project_2019517/moody2026_pilot/retrain/nexttoken_cache \
    --val_size 100
EXIT=$?

chmod -R 600 /scratch/project_2019517/moody2026_pilot/retrain/nexttoken_cache/* 2>/dev/null
echo "SPECTRAL_CACHE_END: $(date +%s) exit=$EXIT"
exit $EXIT
