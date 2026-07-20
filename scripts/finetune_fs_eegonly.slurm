#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ft_fs
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/ft_fs_%j.log

# Fine-tune SleepFM From-Scratch (FS) encoder for EEG_ONLY / ECG_ONLY / EEG_ECG.
#
# Encoder: encoders/fromscratch_eegonly/SetTransformer/leave_one_out_128_patch_size_640/best.pt
# Configs: config_ft_fs_{EEG_ONLY,ECG_ONLY,EEG_ECG}.yaml (all point to the FS encoder)
# Split:   dataset_split_fromscratch_staging.json (same 270/30/50 as all other models)
#
# Each run creates a unique timestamped folder:
#   checkpoints/finetuned/fromscratch/{EEG_ONLY,ECG_ONLY,EEG_ECG}/run_YYYY-MM-DD_HHMM/fold_0/
# No collision with other pretraining approaches or between runs.

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

CHANNELS=sleepfm/configs/channel_groups.json
SPLIT=sleepfm/configs/dataset_split_fromscratch_staging.json

echo "===== EEG-only fine-tuning ====="
python sleepfm/pipeline/finetune_sleep_staging.py \
  --config_path sleepfm/configs/config_ft_fs_EEG_ONLY.yaml \
  --channel_groups_path ${CHANNELS} \
  --split_path ${SPLIT} \
  --fold 0 \
  --pretrain_approach fromscratch

echo "===== ECG-only fine-tuning ====="
python sleepfm/pipeline/finetune_sleep_staging.py \
  --config_path sleepfm/configs/config_ft_fs_ECG_ONLY.yaml \
  --channel_groups_path ${CHANNELS} \
  --split_path ${SPLIT} \
  --fold 0 \
  --pretrain_approach fromscratch

echo "===== EEG+ECG fine-tuning ====="
python sleepfm/pipeline/finetune_sleep_staging.py \
  --config_path sleepfm/configs/config_ft_fs_EEG_ECG.yaml \
  --channel_groups_path ${CHANNELS} \
  --split_path ${SPLIT} \
  --fold 0 \
  --pretrain_approach fromscratch

echo "========== DONE =========="
