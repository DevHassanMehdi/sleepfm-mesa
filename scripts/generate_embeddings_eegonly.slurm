#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gen_emb_eegonly
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/gen_emb_eegonly_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

MODEL_PATH=/scratch/project_2019517/sleepfm-data/checkpoints/fromscratch_eegonly/SetTransformer/leave_one_out_128_patch_size_640
CHANNELS=sleepfm/configs/channel_groups.json
SPLIT=sleepfm/configs/dataset_split_fromscratch_fold.json

python sleepfm/pipeline/generate_embeddings.py \
  --model_path ${MODEL_PATH} \
  --dataset_name mesa \
  --channel_groups_path ${CHANNELS} \
  --split_path ${SPLIT} \
  --splits train,validation,test \
  --fold 0 \
  --num_workers 8 \
  --batch_size 128

echo "========== DONE =========="
