#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=debug_nsrr
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/debug_nsrr_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PATH=/scratch/project_2019517/miniconda3/bin:/users/hamehdi/.local/share/x86_64/gem/ruby/3.4.0/bin:$PATH
python3 /scratch/project_2019517/claude_arm_build/debug_nsrr_download.py
