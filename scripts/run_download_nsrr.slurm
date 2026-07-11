#!/bin/bash
# Download an NSRR dataset via the nsrr gem (no GPU required).
#
# Usage:
#   sbatch --export=DATASET=mesa scripts/run_download_nsrr.slurm
#   sbatch --export=DATASET=shhs scripts/run_download_nsrr.slurm
#   sbatch --export=DATASET=mros  scripts/run_download_nsrr.slurm
#
# DATASET defaults to "mesa" if not set.

#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=nsrr_download
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/download_%x_%j.log

DATASET="${DATASET:-mesa}"
echo "[INFO] Dataset: ${DATASET}"

cd /users/hamehdi/projects/sleepfm-mesa
export GEM_HOME="/scratch/project_2019517/miniconda3/share/rubygems"
export GEM_PATH="/scratch/project_2019517/miniconda3/share/rubygems"

bash scripts/download_nsrr.sh --dataset "${DATASET}"
