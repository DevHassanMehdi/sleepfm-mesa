#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=mesa_finetune
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/finetune_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

python sleepfm/pipeline/finetune_sleep_staging.py --fold 0
