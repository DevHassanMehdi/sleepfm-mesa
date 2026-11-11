#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpularge
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:02:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=diag_nvsmi
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/diag_nvsmi_%j.log

echo "=== nvidia-smi (basic) ==="
nvidia-smi
echo "=== nvidia-smi query (util/mem/power) ==="
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv
echo "=== nvidia-smi -L ==="
nvidia-smi -L
