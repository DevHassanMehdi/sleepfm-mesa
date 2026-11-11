#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_pretrain_full
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/moody_pretrain_full_%j.log

# Time limit derived from the 3-epoch smoke test's measured pace (job
# 2818642, corrected eval_iter): 747 iterations/epoch, ~107ms/iter pure
# training + negligible validation overhead at eval_iter=5000 -> ~2.26h
# estimated for 100 epochs. 6h gives ~2.65x margin. Resume-safe: if this
# allocation runs out, resubmitting the same script picks up from
# checkpoint.pt automatically (pretrain.py's own --checkpoint_path logic).

umask 077
cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

CKPT_DIR=/scratch/project_2019517/moody2026_pilot/retrain/encoders/fromscratch_eegonly_moody/SetTransformer/leave_one_out_128_patch_size_640
CONFIG=sleepfm/configs/config_pretrain_fromscratch_eegonly_moody.yaml
CHANNELS=sleepfm/configs/channel_groups.json

echo "FULLRUN_START: $(date +%s)"

if [ -f "${CKPT_DIR}/checkpoint.pt" ]; then
    echo "Resuming from ${CKPT_DIR}/checkpoint.pt"
    python sleepfm/pipeline/pretrain.py --config_path ${CONFIG} --channel_groups_path ${CHANNELS} --checkpoint_path ${CKPT_DIR}
else
    echo "Starting fresh pretraining"
    python sleepfm/pipeline/pretrain.py --config_path ${CONFIG} --channel_groups_path ${CHANNELS}
fi
PYEXIT=$?

echo "FULLRUN_END: $(date +%s) exit=$PYEXIT"
exit $PYEXIT
