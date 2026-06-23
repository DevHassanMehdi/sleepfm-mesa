#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fs_staging2
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/fs_staging2_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

CONFIG=sleepfm/configs/config_ft_fromscratch.yaml
CHANNELS=sleepfm/configs/channel_groups.json
STAGE_SPLIT=sleepfm/configs/dataset_split_fromscratch_staging.json

for combo in '["BAS"]' '["BAS","EKG"]' '["BAS","EKG","RESP"]' '["BAS","EKG","RESP","EMG"]'; do
    TAG=$(echo $combo | tr -d '[]"' | tr ',' '_')
    echo ">>>>> MODALITY: ${TAG} <<<<<"
    CFG=sleepfm/configs/config_ft_fs_${TAG}.yaml
    cp ${CONFIG} ${CFG}
    sed -i "s|^channel_like:.*|channel_like: ${combo}|" ${CFG}
    python sleepfm/pipeline/finetune_sleep_staging.py \
      --config_path ${CFG} \
      --channel_groups_path ${CHANNELS} \
      --split_path ${STAGE_SPLIT} \
      --fold 0
done
echo "========== DONE =========="
