#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mesa_preprocess_gapfill
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/preprocess_gapfill_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate

# Targets the full EDF dir (2056 files); EDFToHDF5Converter.convert_multiprocessing()
# skips any file that already has an HDF5 output (os.path.exists check), so this
# only actually processes the 112 subjects missing from hdf5_full -- no need to
# stage a separate input directory.
python sleepfm/preprocessing/preprocessing.py \
    --root_dir data/mesa/edf \
    --target_dir data/mesa/hdf5_full \
    --num_threads 8 \
    --num_files -1 \
    --resample_rate 128
