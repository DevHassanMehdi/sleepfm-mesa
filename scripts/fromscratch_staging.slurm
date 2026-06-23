#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fs_staging
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/fs_staging_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

MODEL=/scratch/project_2019517/sleepfm-data/checkpoints/fromscratch/SetTransformer/leave_one_out_128_patch_size_640
EMBED_SPLIT=sleepfm/configs/dataset_split_fromscratch_fold.json
STAGE_SPLIT=sleepfm/configs/dataset_split_fromscratch_staging.json
CONFIG=sleepfm/configs/config_ft_fromscratch.yaml
CHANNELS=sleepfm/configs/channel_groups.json

echo "========== STEP 1: GENERATE EMBEDDINGS FROM FROM-SCRATCH ENCODER =========="
python sleepfm/pipeline/generate_embeddings.py \
  --model_path ${MODEL} \
  --dataset_name mesa \
  --split_path ${EMBED_SPLIT} \
  --splits test \
  --fold 0 \
  --batch_size 128 \
  --num_workers 8

echo "========== STEP 2: MODALITY ABLATION (train 270, test held-out 50) =========="
for combo in '["BAS"]' '["BAS","EKG"]' '["BAS","EKG","RESP"]' '["BAS","EKG","RESP","EMG"]'; do
    TAG=$(echo $combo | tr -d '[]"' | tr ',' '_')
    echo ">>>>> MODALITY: ${TAG} <<<<<"
    # set channel_like in a per-combo config copy
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
