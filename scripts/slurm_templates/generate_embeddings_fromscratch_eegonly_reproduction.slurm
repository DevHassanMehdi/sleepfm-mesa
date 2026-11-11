#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=embed_fromscratch_eegonly_repro
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/embed_fromscratch_eegonly_reproduction_%j.log

# Embeddings from the FRESH reproduction encoder's best.pt, for the full
# 1944-subject cohort (only the 50 historical test subjects are actually
# needed downstream, but reusing the same full-cohort split file keeps
# this consistent with every other embeddings job this session).

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

echo "EMBED_START: $(date +%s)"
python sleepfm/pipeline/generate_embeddings.py \
    --model_path /scratch/project_2019517/sleepfm-data/checkpoints/encoders/fromscratch_eegonly_reproduction/SetTransformer/leave_one_out_128_patch_size_640 \
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
