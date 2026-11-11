#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=check_kaggle_net
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/check_kaggle_net_%j.log

echo "=== compute node: $(hostname) ==="
echo "=== DNS ==="
getent hosts kaggle.com || echo "DNS FAILED"
getent hosts www.kaggle.com || echo "DNS FAILED (www)"
echo "=== curl kaggle.com ==="
curl -sS -o /dev/null -w "HTTP %{http_code}, connect_time=%{time_connect}s, total_time=%{time_total}s\n" --max-time 15 https://www.kaggle.com
echo "curl exit code: $?"
echo "=== curl kaggle API endpoint ==="
curl -sS -o /dev/null -w "HTTP %{http_code}, connect_time=%{time_connect}s, total_time=%{time_total}s\n" --max-time 15 https://www.kaggle.com/api/v1/datasets/list
echo "curl exit code: $?"
