#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_nextwindow_ft
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/finetune_nextwindow_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

MODEL=/scratch/project_2019517/sleepfm-data/checkpoints/nextwindow/SetTransformer/nextwindow_128_patch_size_640
EMBED_SPLIT=sleepfm/configs/dataset_split_fromscratch_fold.json
STAGE_SPLIT=sleepfm/configs/dataset_split_fromscratch_staging.json
BASE_CONFIG=sleepfm/configs/config_ft_fromscratch.yaml
CHANNELS=sleepfm/configs/channel_groups.json

echo "========== STEP 1: GENERATE EMBEDDINGS FROM NEXTWINDOW ENCODER =========="
python sleepfm/pipeline/generate_embeddings.py \
  --model_path ${MODEL} \
  --dataset_name mesa \
  --split_path ${EMBED_SPLIT} \
  --splits test \
  --fold 0 \
  --batch_size 128 \
  --num_workers 8

echo "========== STEP 2: MODALITY ABLATION (6 configs) =========="
for combo in '["BAS"]' '["BAS","EKG"]' '["BAS","EKG","RESP"]' '["BAS","EKG","RESP","EMG"]' '["EEG_ONLY"]' '["EKG"]'; do
    TAG=$(echo $combo | tr -d '[]"' | tr ',' '_')
    echo ">>>>> MODALITY: ${TAG} <<<<<"
    CFG=sleepfm/configs/config_ft_nextwindow_${TAG}.yaml
    cp ${BASE_CONFIG} ${CFG}
    sed -i "s|^channel_like:.*|channel_like: ${combo}|" ${CFG}
    sed -i "s|^model_path:.*|model_path: ${MODEL}|" ${CFG}
    python sleepfm/pipeline/finetune_sleep_staging.py \
      --config_path ${CFG} \
      --channel_groups_path ${CHANNELS} \
      --split_path ${STAGE_SPLIT} \
      --fold 0
    echo "========== DONE ${TAG} =========="
done

echo "========== ALL NEXTWINDOW FINE-TUNING DONE =========="
