#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pretrain_fromscratch_eegonly_repro
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/pretrain_fromscratch_eegonly_reproduction_%j.log

# Fresh, from-random-init reproduction of the original 350-subject
# From-Scratch EEG_ONLY encoder. New output path (checkpoints/encoders/
# fromscratch_eegonly_reproduction/...) -- does not touch the original
# June encoder. Estimated ~3h based on full-cohort's GH200 benchmark
# (10.1 min/epoch @ 1944 subjects) scaled by the 350/1944 subject-count
# ratio (~1.8 min/epoch @ 350 subjects x 100 epochs). 6h time limit
# gives generous margin. No early-stopping in pretrain.py -- will run
# the full 100 configured epochs regardless.

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehd/projects/sleepfm-mesa/sleepfm:/users/hamehd/projects/sleepfm-mesa

echo "REPRO_START: $(date +%s)"
python sleepfm/pipeline/pretrain.py \
    --config_path sleepfm/configs/config_pretrain_fromscratch_eegonly_reproduction.yaml \
    --channel_groups_path sleepfm/configs/channel_groups.json
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: pretrain.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "REPRO_END: $(date +%s)"
