#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpularge
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gen_emb_fullcohort_v2
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/gen_emb_fullcohort_v2_%x_%j.log

# Usage: sbatch generate_embeddings_fullcohort_v2.slurm <MODEL_PATH> <JOBTAG>
# Generates embeddings for ALL 2056 subjects in ONE pass: fold_0's
# train+validation+test partition (1234+411+411) sums to the full cohort,
# and embeddings are a per-subject function of the frozen encoder --
# independent of which fold a subject is assigned to. These cached
# embeddings are reused by every fold's fine-tuning run.
set -e
MODEL_PATH=${1:?Usage: sbatch generate_embeddings_fullcohort_v2.slurm <MODEL_PATH> <JOBTAG>}
JOBTAG=${2:?Usage: sbatch generate_embeddings_fullcohort_v2.slurm <MODEL_PATH> <JOBTAG>}

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

echo "EMBED_START: $(date +%s)  JOBTAG=${JOBTAG}  MODEL_PATH=${MODEL_PATH}"
python sleepfm/pipeline/generate_embeddings.py \
    --model_path "${MODEL_PATH}" \
    --dataset_name mesa \
    --channel_groups_path sleepfm/configs/channel_groups.json \
    --split_id fold5_v1 \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --splits train,validation,test \
    --fold 0 \
    --num_workers 8 \
    --batch_size 32
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: generate_embeddings.py exited with code $PYEXIT"
    exit $PYEXIT
fi
echo "EMBED_END: $(date +%s)"
