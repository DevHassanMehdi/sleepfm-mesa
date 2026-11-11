#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_preprocess_timetest
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/preprocess_timetest_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate

mkdir -p /scratch/project_2019517/sleepfm-data/mesa/hdf5_full_timetest

echo "TIMETEST_START: $(date +%s)"
time python sleepfm/preprocessing/preprocessing.py \
    --root_dir data/mesa/edf \
    --target_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full_timetest \
    --num_threads 1 \
    --num_files 20 \
    --resample_rate 128
echo "TIMETEST_END: $(date +%s)"
