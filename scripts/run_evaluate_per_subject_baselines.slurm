#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eval_ps_baselines
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/eval_ps_baselines_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

python scripts/evaluate_per_subject_baselines.py \
  --models BIOT LaBraM MOMENT SensorLM \
  --modalities EEG_ONLY ECG_ONLY EEG_ECG \
  --fold_key fold_0 \
  --num_workers 4

echo "========== DONE =========="
