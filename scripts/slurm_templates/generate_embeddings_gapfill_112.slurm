#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=embed_pub_gapfill112
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/embed_pub_gapfill112_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE

python sleepfm/pipeline/generate_embeddings.py \
    --model_path sleepfm/checkpoints/model_base \
    --dataset_name mesa \
    --split_path /scratch/project_2019517/claude_arm_build/split_112_new_subjects_only.json \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --splits train \
    --num_workers 8 \
    --batch_size 32 \
    --fold 0

echo "========== EMBEDDING GAPFILL DONE =========="
