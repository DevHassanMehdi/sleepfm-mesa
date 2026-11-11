#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_nexttoken_pretrain
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moody_nexttoken_pretrain_%j.log

# Full Next-Token pretraining on Moody (100 epochs, patience 10, same
# recipe as MESA's config_pretrain_nexttoken_eegonly_fullcohort.yaml --
# only paths changed). Resume: pretrain_nexttoken.py's output path is
# fully deterministic (save_path + fixed subpath, no timestamp), and it
# auto-resumes from checkpoint.pt if that file already exists there --
# no pointer-file hack needed, just resubmit this same script unchanged
# if the time limit is hit before patience triggers.
# 3-epoch smoke test (job 2979708) already confirmed validation moves
# (3.4029 -> 3.3882 -> 3.3504) before this submission.

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "PRETRAIN_START: $(date +%s)"
python sleepfm/pipeline/pretrain_nexttoken.py \
    --config_path sleepfm/configs/config_pretrain_nexttoken_moody.yaml
EXIT=$?
echo "PRETRAIN_END: $(date +%s) exit=$EXIT"
exit $EXIT
