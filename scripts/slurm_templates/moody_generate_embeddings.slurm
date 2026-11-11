#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=moody_embed
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moody_embed_%x_%j.log

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

# Usage: sbatch moody_generate_embeddings.slurm <ENCODER_NAME>
ENCODER_NAME=${1:?Usage: sbatch moody_generate_embeddings.slurm <ENCODER_NAME>}
MODEL_PATH=/scratch/project_2019517/moody2026_pilot/encoders_for_embeddings/${ENCODER_NAME}

echo "START: $(date +%s) encoder=${ENCODER_NAME}"
echo ">>> CHECKPOINT PATH: ${MODEL_PATH}/best.pt"
echo ">>> CONFIG modality_types: $(python3 -c "import json; print(json.load(open('${MODEL_PATH}/config.json'))['modality_types'])")"
md5sum "${MODEL_PATH}/best.pt"

python sleepfm/pipeline/generate_embeddings.py \
    --model_path "${MODEL_PATH}" \
    --dataset_name i0006 \
    --split_path /scratch/project_2019517/moody2026_pilot/splits/dataset_split_moody_i0006_testonly.json \
    --hdf5_dir /scratch/project_2019517/moody2026_pilot/hdf5_included \
    --splits test \
    --fold 0
EXIT=$?

echo ">>> embeddings output listing:"
ls "${MODEL_PATH}/i0006" 2>/dev/null | wc -l
echo "END: $(date +%s) exit=$EXIT"
exit $EXIT
