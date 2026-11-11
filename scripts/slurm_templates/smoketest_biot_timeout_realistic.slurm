#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpularge
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:08:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smoketest_biot_timeout_realistic
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/smoketest_biot_timeout_realistic_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

echo "SMOKETEST_START: $(date +%s)"
python scripts/finetune_biot.py \
    --modality EEG_ONLY \
    --epochs 100 \
    --patience 100
PYEXIT=$?
echo "PYEXIT=$PYEXIT"
echo "SMOKETEST_END: $(date +%s)"
