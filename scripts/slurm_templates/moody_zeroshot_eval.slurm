#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_zeroshot
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moody_zeroshot_%j.log

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
sys_path_check=$(python3 -c "import sys; print('/scratch/project_2019517/BIOT' in sys.path)")

echo "START: $(date +%s)"
python3 /scratch/project_2019517/claude_arm_build/moody_zeroshot_eval.py
EXIT=$?
echo "END: $(date +%s) exit=$EXIT"
exit $EXIT
