#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_preprocess
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/preprocess_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

python sleepfm/preprocessing/preprocessing.py \
    --root_dir data/mesa/edf \
    --target_dir data/mesa/hdf5 \
    --num_threads 8 \
    --num_files -1 \
    --resample_rate 128
