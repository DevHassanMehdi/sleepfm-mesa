#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_embed_retrained
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moody_embed_retrained_%x_%j.log

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

# Usage: sbatch moody_generate_embeddings_retrained.slurm <MODEL_PATH>
# MODEL_PATH is the full path to a SetTransformer/leave_one_out_.../ dir
# containing best.pt + config.json -- either the retrained Moody encoder
# itself, or a modality_types-only copy of it (e.g. for EEG3_ONLY).
# Already entirely inside the Moody area, owner-only -- unlike the
# zero-shot evaluation turn, no copy-out-of-the-shared-tree is needed
# here since this encoder was trained ON Moody data and already lives
# under /scratch/project_2019517/moody2026_pilot/retrain/.
MODEL_PATH=${1:?Usage: sbatch moody_generate_embeddings_retrained.slurm <MODEL_PATH>}

echo "START: $(date +%s) model_path=${MODEL_PATH}"
echo ">>> CHECKPOINT PATH: ${MODEL_PATH}/best.pt"
echo ">>> CONFIG modality_types: $(python3 -c "import json; print(json.load(open('${MODEL_PATH}/config.json'))['modality_types'])")"
md5sum "${MODEL_PATH}/best.pt"
date -r "${MODEL_PATH}/best.pt" 2>/dev/null || stat "${MODEL_PATH}/best.pt"

python sleepfm/pipeline/generate_embeddings.py \
    --model_path "${MODEL_PATH}" \
    --dataset_name i0006 \
    --split_path /scratch/project_2019517/moody2026_pilot/splits/dataset_split_moody_i0006_embeddings_all.json \
    --hdf5_dir /scratch/project_2019517/moody2026_pilot/hdf5_included \
    --splits all \
    --fold 0
EXIT=$?

echo ">>> embeddings output listing:"
ls "${MODEL_PATH}/i0006" 2>/dev/null | wc -l
echo "END: $(date +%s) exit=$EXIT"
exit $EXIT
