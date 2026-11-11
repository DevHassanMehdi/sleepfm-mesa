#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=1-12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pretrain_nexttoken_eegonly_fc
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/pretrain_nexttoken_eegonly_fullcohort_real_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "REALRUN_START: $(date +%s) (job $SLURM_JOB_ID)"

python sleepfm/pipeline/pretrain_nexttoken.py \
    --config_path sleepfm/configs/config_pretrain_nexttoken_eegonly_fullcohort.yaml

echo "REALRUN_END: $(date +%s) (job $SLURM_JOB_ID)"
