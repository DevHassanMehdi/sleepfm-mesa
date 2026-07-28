#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=30:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_preprocess
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/preprocess_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate

mkdir -p /scratch/project_2019517/sleepfm-data/mesa/hdf5_full

python sleepfm/preprocessing/preprocessing.py \
    --root_dir data/mesa/edf \
    --target_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --num_threads 1 \
    --num_files -1 \
    --resample_rate 128
