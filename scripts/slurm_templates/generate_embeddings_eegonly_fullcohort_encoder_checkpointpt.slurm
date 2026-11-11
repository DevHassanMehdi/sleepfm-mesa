#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=embed_eegonly_fullcohort_checkpointpt
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/embed_eegonly_fullcohort_checkpointpt_%j.log

# Same encoder, same subject set as the existing best.pt-derived embeddings
# cache -- only the checkpoint file differs (final epoch 99 instead of the
# training-loss-selected epoch 28). Writes to model_path/mesa_checkpoint/
# (and mesa_5min_agg_checkpoint/), never touching the existing mesa/ cache.

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

echo "EMBED_START: $(date +%s)"
python sleepfm/pipeline/generate_embeddings.py \
    --model_path /scratch/project_2019517/sleepfm-data/checkpoints/encoders/fromscratch_eegonly_fullcohort/SetTransformer/leave_one_out_128_patch_size_640 \
    --dataset_name mesa \
    --split_path sleepfm/configs/embeddings_split_full_cohort_eegonly_fullcohort.json \
    --splits all \
    --fold 0 \
    --num_workers 8 \
    --batch_size 32 \
    --checkpoint_file checkpoint.pt
echo "EMBED_END: $(date +%s)"
