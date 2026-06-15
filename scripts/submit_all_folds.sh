#!/bin/bash
for fold in {0..9}; do
    cat > /tmp/finetune_fold${fold}.slurm << SLURM
#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=finetune_f${fold}
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/finetune_fold${fold}_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa
export HDF5_USE_FILE_LOCKING=FALSE
source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
conda activate sleepfm_env

python sleepfm/pipeline/finetune_sleep_staging.py --fold ${fold}
SLURM
    sbatch /tmp/finetune_fold${fold}.slurm
    echo "Submitted fold ${fold}"
done
