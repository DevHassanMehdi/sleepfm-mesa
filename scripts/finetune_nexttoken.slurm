#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ft_nexttoken
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/ft_nexttoken_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

CHANNELS=sleepfm/configs/channel_groups.json
SPLIT=sleepfm/configs/dataset_split_fromscratch_staging.json

echo "===== EEG-only fine-tuning ====="
python sleepfm/pipeline/finetune_sleep_staging.py \
  --config_path sleepfm/configs/config_ft_nexttoken_EEG_ONLY.yaml \
  --channel_groups_path ${CHANNELS} \
  --split_path ${SPLIT} \
  --fold 0 \
  --pretrain_approach nexttoken

echo "===== ECG-only fine-tuning ====="
python sleepfm/pipeline/finetune_sleep_staging.py \
  --config_path sleepfm/configs/config_ft_nexttoken_ECG_ONLY.yaml \
  --channel_groups_path ${CHANNELS} \
  --split_path ${SPLIT} \
  --fold 0 \
  --pretrain_approach nexttoken

echo "===== EEG+ECG fine-tuning ====="
python sleepfm/pipeline/finetune_sleep_staging.py \
  --config_path sleepfm/configs/config_ft_nexttoken_EEG_ECG.yaml \
  --channel_groups_path ${CHANNELS} \
  --split_path ${SPLIT} \
  --fold 0 \
  --pretrain_approach nexttoken

echo "========== DONE =========="
