#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eval_spectral
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/eval_spectral_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

for DIR in SleepEventLSTMClassifier_mesa_labels_EEG_ONLY SleepEventLSTMClassifier_mesa_labels_EKG SleepEventLSTMClassifier_mesa_labels_EEG_ONLY_EKG; do
    echo "===== Evaluating ${DIR} ====="
    python sleepfm/pipeline/evaluate_sleep_staging.py \
        --output_path /scratch/project_2019517/sleepfm-data/checkpoints/${DIR} \
        --split test --fold 0

    echo "===== Metrics for ${DIR} ====="
    python scripts/compute_metrics.py \
        --checkpoint_dir /scratch/project_2019517/sleepfm-data/checkpoints/${DIR} \
        --n_folds 1
done
echo "========== DONE =========="
