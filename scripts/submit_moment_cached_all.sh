#!/bin/bash
# Submit MOMENT cached-embedding pipeline for all 7 modality configurations.
#
# For each modality:
#   1. Submit run_moment_embed.slurm (GPU, ~2-4h): generates embeddings once
#   2. Submit run_moment_cached.slurm (GPU, ~1h): trains head, evaluates
#      with --dependency=afterok:<embed_job> so it starts only after embeddings finish
#
# If embeddings already exist (done markers present), the embed job exits immediately.
# Safe to resubmit the whole script; done markers prevent redundant work.
#
# Usage: bash scripts/submit_moment_cached_all.sh
set -e
cd /users/hamehdi/projects/sleepfm-mesa
mkdir -p logs

MODALITIES="EEG_ONLY ECG_ONLY EEG_ECG"

for MODALITY in $MODALITIES; do
    echo "Submitting MOMENT cached pipeline for ${MODALITY}"

    EMBED_JOB=$(sbatch --parsable \
        --job-name="moment_embed_${MODALITY}" \
        --output="logs/moment_embed_${MODALITY}_%j.log" \
        scripts/run_moment_embed.slurm ${MODALITY})
    echo "  embed job: ${EMBED_JOB}"

    TRAIN_JOB=$(sbatch --parsable \
        --dependency=afterok:${EMBED_JOB} \
        --job-name="moment_cached_${MODALITY}" \
        --output="logs/moment_cached_${MODALITY}_%j.log" \
        scripts/run_moment_cached.slurm ${MODALITY})
    echo "  train job: ${TRAIN_JOB} (depends on ${EMBED_JOB})"
done
