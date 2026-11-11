#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpularge
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:03:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smoketest_ft_ss_resume
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/smoketest_ft_ss_resume_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

echo "SMOKETEST_START: $(date +%s)"
python sleepfm/pipeline/finetune_sleep_staging.py \
    --checkpoint_path /scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__EEG_ONLY__spectral__fold5_v1__resumetest-9999/fold_0 \
    --channel_groups_path sleepfm/configs/channel_groups.json \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --split_id fold5_v1 \
    --fold 0 \
    --pretrain_approach spectral
PYEXIT=$?
echo "PYEXIT=$PYEXIT"
echo "SMOKETEST_END: $(date +%s)"
