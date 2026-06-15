#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=mesa_embeddings
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/embeddings_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env
export HDF5_USE_FILE_LOCKING=FALSE

python sleepfm/pipeline/generate_embeddings.py \
    --model_path sleepfm/checkpoints/model_base \
    --dataset_name mesa \
    --splits train,validation,test \
    --num_workers 4 \
    --batch_size 32 \
    --fold 0
