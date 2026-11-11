#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_hdf5_dir_override
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/test_hdf5_dir_override_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE

python3 << 'PYEOF'
import sys, os
sys.path.insert(0, "/users/hamehdi/projects/sleepfm-mesa/scripts")

FULL_COHORT_HDF5 = "/scratch/project_2019517/sleepfm-data/mesa/hdf5_full"
SPLIT_PATH = "/users/hamehdi/projects/sleepfm-mesa/sleepfm/configs/dataset_split_fold10_v1.json"

def unique_subjects(ds):
    return len(set(os.path.basename(t[0]) for t in ds.index))

print("=== BIOTSleepDataset ===")
from biot_dataset import BIOTSleepDataset, HDF5_DIR as BIOT_DEFAULT
print("module default HDF5_DIR:", BIOT_DEFAULT)
ds = BIOTSleepDataset(SPLIT_PATH, "train", "EEG_ONLY", fold_key="fold_0", hdf5_dir=FULL_COHORT_HDF5)
print(f"full-cohort override: len(dataset)={len(ds)} (epoch-level), unique subjects={unique_subjects(ds)}")
ds_old = BIOTSleepDataset(SPLIT_PATH, "train", "EEG_ONLY", fold_key="fold_0")
print(f"no override (350-subj default): len(dataset)={len(ds_old)}, unique subjects={unique_subjects(ds_old)}")

print()
print("=== LaBraMSleepDataset ===")
from labram_dataset import LaBraMSleepDataset, HDF5_DIR as LABRAM_DEFAULT
print("module default HDF5_DIR:", LABRAM_DEFAULT)
ds2 = LaBraMSleepDataset(SPLIT_PATH, "train", "EEG_ONLY", fold_key="fold_0", hdf5_dir=FULL_COHORT_HDF5)
print(f"full-cohort override: len(dataset)={len(ds2)} (epoch-level), unique subjects={unique_subjects(ds2)}")
ds2_old = LaBraMSleepDataset(SPLIT_PATH, "train", "EEG_ONLY", fold_key="fold_0")
print(f"no override (350-subj default): len(dataset)={len(ds2_old)}, unique subjects={unique_subjects(ds2_old)}")

print()
print("=== SensorLMSleepDataset ===")
from sensorlm_dataset import SensorLMSleepDataset, HDF5_DIR as SENSORLM_DEFAULT
print("module default HDF5_DIR:", SENSORLM_DEFAULT)
ds3 = SensorLMSleepDataset(SPLIT_PATH, "train", "EEG_ONLY", "fold_0", hdf5_dir=FULL_COHORT_HDF5)
print(f"full-cohort override: len(dataset)={len(ds3)} (epoch-level), unique subjects={unique_subjects(ds3)}")
ds3_old = SensorLMSleepDataset(SPLIT_PATH, "train", "EEG_ONLY", "fold_0")
print(f"no override (350-subj default): len(dataset)={len(ds3_old)}, unique subjects={unique_subjects(ds3_old)}")

print()
print("DRYRUN_SUMMARY_OK")
PYEOF
echo "DRYRUN_DONE"
