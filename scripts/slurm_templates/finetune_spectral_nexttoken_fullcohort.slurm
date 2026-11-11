#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpularge
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ft_ss_fullcohort
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/ft_ss_fullcohort_%x_%j.log

# Usage: sbatch --time=<H:MM:SS> finetune_spectral_nexttoken_fullcohort.slurm \
#            <CONFIG_YAML> <PRETRAIN_APPROACH> <FOLD> <CANONICAL_MODALITY> <TIMESTAMP>
# CANONICAL_MODALITY is the shared-naming-scheme modality (EEG_ONLY / ECG_ONLY /
# EEG_ECG), used only to construct the checkpoint_dir path for the eval/metrics
# steps -- experiment_paths.py's canonical_modality() maps the config's raw
# channel_like ("EKG" / "EEG_ONLY_EKG") to this same string internally.
set -e
CONFIG_YAML=${1:?Usage: <CONFIG_YAML> <PRETRAIN_APPROACH> <FOLD> <CANONICAL_MODALITY> <TIMESTAMP>}
PRETRAIN_APPROACH=${2:?Usage: <CONFIG_YAML> <PRETRAIN_APPROACH> <FOLD> <CANONICAL_MODALITY> <TIMESTAMP>}
FOLD=${3:?Usage: <CONFIG_YAML> <PRETRAIN_APPROACH> <FOLD> <CANONICAL_MODALITY> <TIMESTAMP>}
CANONICAL_MODALITY=${4:?Usage: <CONFIG_YAML> <PRETRAIN_APPROACH> <FOLD> <CANONICAL_MODALITY> <TIMESTAMP>}
TIMESTAMP=${5:?Usage: <CONFIG_YAML> <PRETRAIN_APPROACH> <FOLD> <CANONICAL_MODALITY> <TIMESTAMP>}

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export TMPDIR=/scratch/project_2019517/tmp
mkdir -p "$TMPDIR"

CHECKPOINT_DIR="/scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__${CANONICAL_MODALITY}__${PRETRAIN_APPROACH}__fold5_v1__${TIMESTAMP}"

echo "FT_START: $(date +%s)  CONFIG=${CONFIG_YAML}  APPROACH=${PRETRAIN_APPROACH}  FOLD=${FOLD}  MODALITY=${CANONICAL_MODALITY}  TS=${TIMESTAMP}"
echo "EXPECTED CHECKPOINT_DIR: ${CHECKPOINT_DIR}"

python sleepfm/pipeline/finetune_sleep_staging.py \
    --config_path "${CONFIG_YAML}" \
    --channel_groups_path sleepfm/configs/channel_groups.json \
    --hdf5_dir /scratch/project_2019517/sleepfm-data/mesa/hdf5_full \
    --split_id fold5_v1 \
    --fold "${FOLD}" \
    --pretrain_approach "${PRETRAIN_APPROACH}" \
    --timestamp "${TIMESTAMP}"
PYEXIT=$?
if [ $PYEXIT -ne 0 ]; then
    echo "FATAL: finetune_sleep_staging.py exited with code $PYEXIT"
    exit $PYEXIT
fi

echo "===== Evaluating (fold ${FOLD}) ====="
python sleepfm/pipeline/evaluate_sleep_staging.py \
    --config_path "${CONFIG_YAML}" \
    --channel_groups_path sleepfm/configs/channel_groups.json \
    --output_path "${CHECKPOINT_DIR}" \
    --split test \
    --fold "${FOLD}"
EVALEXIT=$?
if [ $EVALEXIT -ne 0 ]; then
    echo "FATAL: evaluate_sleep_staging.py exited with code $EVALEXIT"
    exit $EVALEXIT
fi

echo "===== Computing metrics (fold ${FOLD}) ====="
python scripts/compute_metrics.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --split test \
    --fold "${FOLD}"
METRICSEXIT=$?
if [ $METRICSEXIT -ne 0 ]; then
    echo "FATAL: compute_metrics.py exited with code $METRICSEXIT"
    exit $METRICSEXIT
fi
echo "FT_END: $(date +%s)"
