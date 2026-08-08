#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=dl_mesa_straggler
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/download_mesa_straggler_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PATH=/scratch/project_2019517/miniconda3/bin:/users/hamehdi/.local/share/x86_64/gem/ruby/3.4.0/bin:$PATH

python scripts/download_mesa.py --subject-list /scratch/project_2019517/claude_arm_build/mesa_last_straggler.txt
echo "========== MESA STRAGGLER DOWNLOAD DONE =========="
