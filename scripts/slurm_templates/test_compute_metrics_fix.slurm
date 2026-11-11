#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=test_compute_metrics
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/test_compute_metrics_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
export PYTHONPATH=/users/hamehdi/projects/sleepfm-mesa/sleepfm:/users/hamehdi/projects/sleepfm-mesa

TMPDIR=/tmp/compute_metrics_fix_test
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

echo "=== TEST 1: --fold 0 against a real fold_0 checkpoint (regression check vs known-good run) ==="
CKPT1="$TMPDIR/sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-07-28_1623"
cp -r /scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-07-28_1623 "$CKPT1"
python scripts/compute_metrics.py --checkpoint_dir "$CKPT1" --split test --fold 0
echo "TEST1_EXIT_CODE: $?"

echo
echo "=== TEST 2: --fold 3 against a checkpoint dir containing only fold_3/ (simulates the real scenario: one non-zero fold, own unique dir, valid run_name) ==="
CKPT2="$TMPDIR/sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-07-28_9999"
mkdir -p "$CKPT2/fold_3"
cp /scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-07-28_1623/fold_0/*.pickle "$CKPT2/fold_3/"
cp /scratch/project_2019517/sleepfm-data/checkpoints/full_cohort/sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-07-28_1623/fold_0/config.json "$CKPT2/"
python scripts/compute_metrics.py --checkpoint_dir "$CKPT2" --split test --fold 3
echo "TEST2_EXIT_CODE: $?"

echo
echo "=== TEST 3: --fold 3 against the SAME dir but querying wrong fold (fold_0 doesn't exist here) -- must fail loudly, not silently succeed ==="
python scripts/compute_metrics.py --checkpoint_dir "$CKPT2" --split test --fold 0
echo "TEST3_EXIT_CODE: $?"

rm -rf "$TMPDIR"
echo "DONE"
