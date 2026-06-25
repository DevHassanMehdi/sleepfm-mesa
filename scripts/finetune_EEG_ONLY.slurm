#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ft_EEG_ONLY
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/ft_EEG_ONLY_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

python sleepfm/pipeline/finetune_sleep_staging.py   --config_path sleepfm/configs/config_ft_fs_EEG_ONLY.yaml   --channel_groups_path sleepfm/configs/channel_groups.json   --split_path sleepfm/configs/fromscratch_heldout_flat.json   --fold 0

echo "========== DONE EEG_ONLY =========="
