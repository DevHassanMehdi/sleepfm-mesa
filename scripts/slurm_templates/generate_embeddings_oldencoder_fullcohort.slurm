#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=embed_oldencoder_fullcohort
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/embed_oldencoder_fullcohort_%j.log

# Experiment A: embeddings for the full 1944-subject cohort generated from
# the ORIGINAL 350-subject fromscratch_eegonly encoder's best.pt (weights
# untouched since 2026-06-24; only its config.json's data_path was fixed to
# a relative symlink path, same as the full-cohort encoders' convention).
# Output lands in this encoder's own model_path/mesa/ -- a location with no
# prior embeddings, so nothing existing is touched.

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

echo "EMBED_START: $(date +%s)"
python sleepfm/pipeline/generate_embeddings.py \
    --model_path /scratch/project_2019517/sleepfm-data/checkpoints/encoders/fromscratch_eegonly/SetTransformer/leave_one_out_128_patch_size_640 \
    --dataset_name mesa \
    --split_path sleepfm/configs/embeddings_split_full_cohort_eegonly_fullcohort.json \
    --splits all \
    --fold 0 \
    --num_workers 8 \
    --batch_size 32
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: generate_embeddings.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "EMBED_END: $(date +%s)"
