#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=longrun
#SBATCH --time=220:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=dl_shhs
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/download_shhs_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PATH=/scratch/project_2019517/miniconda3/bin:/users/hamehdi/.local/share/x86_64/gem/ruby/3.4.0/bin:$PATH

python scripts/download_shhs.py --subjects 10000
echo "========== SHHS DOWNLOAD DONE =========="
