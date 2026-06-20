#!/bin/bash
# Usage: bash scripts/submit_experiment.sh <config_filename>
# e.g.   bash scripts/submit_experiment.sh config_ft_BAS.yaml
CONFIG=$1
if [ -z "$CONFIG" ]; then echo "Usage: bash scripts/submit_experiment.sh <config_filename>"; exit 1; fi
CONFIG_PATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm/configs/${CONFIG}
TAG=$(echo "$CONFIG" | sed 's/config_ft_//; s/.yaml//')

for fold in {0..9}; do
    cat > /tmp/ft_${TAG}_fold${fold}.slurm << SLURM
#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ft_${TAG}_f${fold}
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/ft_${TAG}_fold${fold}_%j.log
cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env
python sleepfm/pipeline/finetune_sleep_staging.py --config_path ${CONFIG_PATH} --fold ${fold}
SLURM
    sbatch /tmp/ft_${TAG}_fold${fold}.slurm
done
echo "Submitted 10 folds for ${TAG}"
