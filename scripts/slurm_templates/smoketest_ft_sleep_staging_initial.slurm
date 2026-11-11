#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpularge
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smoketest_ft_ss_initial
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/smoketest_ft_ss_initial_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

echo "SMOKETEST_START: $(date +%s)"
python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path sleepfm/configs/config_ft_spectral_eegonly_fullcohort_EEG_ONLY.yaml \
    --channel_groups_path sleepfm/configs/channel_groups.json \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --split_id fold5_v1 \
    --fold 0 \
    --pretrain_approach spectral \
    --timestamp resumetest-9999
PYEXIT=$?
echo "PYEXIT=$PYEXIT"
echo "SMOKETEST_END: $(date +%s)"
