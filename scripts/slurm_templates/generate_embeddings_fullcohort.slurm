#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=generate_embeddings_fullcohort
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/generate_embeddings_fullcohort_%x_%j.log

# Usage: sbatch generate_embeddings_fullcohort.slurm <MODEL_PATH>
set -e
MODEL_PATH=${1:?Usage: sbatch generate_embeddings_fullcohort.slurm <MODEL_PATH>}

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "EMBEDDINGS_START: $(date +%s)  MODEL_PATH=${MODEL_PATH}"
python sleepfm/pipeline/generate_embeddings.py \
    --model_path "${MODEL_PATH}" \
    --dataset_name mesa \
    --split_id fold5_v1 \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --splits train,validation,test \
    --fold 0 \
    --num_workers 8 \
    --batch_size 128
echo "EMBEDDINGS_END: $(date +%s)"
