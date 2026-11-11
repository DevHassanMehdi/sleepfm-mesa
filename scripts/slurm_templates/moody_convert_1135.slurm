#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=moody_convert_1135
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/moody_convert_1135_%j.log

umask 077
cd /users/hamehdi/projects/sleepfm-mesa
source /scratch/project_2019517/kaggle_venv_py312/bin/activate

PHYSIO_DIR=/scratch/project_2019517/moody2026_pilot/raw_large/physiological_data/I0006/physiological_data/I0006
ANNOT_DIR=/scratch/project_2019517/moody2026_pilot/raw_large/human_annotations/I0006/human_annotations/I0006
FILE_LIST=/scratch/project_2019517/moody2026_pilot/.included_1135_filelist.txt
HDF5_OUT=/scratch/project_2019517/moody2026_pilot/hdf5_included
LABELS_OUT=/scratch/project_2019517/moody2026_pilot/labels_included

mkdir -p "$HDF5_OUT" "$LABELS_OUT"
chmod 700 "$HDF5_OUT" "$LABELS_OUT"

# Build EDF-suffixed and annotation-suffixed file lists from the plain
# subject-ID list (one subject ID per line, no extension).
PHYSIO_FILELIST=$(mktemp)
ANNOT_FILELIST=$(mktemp)
while read -r sid; do
    echo "${sid}.edf" >> "$PHYSIO_FILELIST"
    echo "${sid}_expert_annotations.edf" >> "$ANNOT_FILELIST"
done < "$FILE_LIST"
chmod 600 "$PHYSIO_FILELIST" "$ANNOT_FILELIST"

echo "=== HDF5 conversion START: $(date +%s) ==="
python scripts/moody_to_hdf5.py \
    --edf_dir "$PHYSIO_DIR" \
    --target_dir "$HDF5_OUT" \
    --file_list "$PHYSIO_FILELIST" \
    --num_workers 16
HDF5_EXIT=$?
echo "=== HDF5 conversion END: $(date +%s) exit=$HDF5_EXIT ==="
if [ $HDF5_EXIT -ne 0 ]; then
    echo "FATAL: HDF5 conversion failed"
    exit $HDF5_EXIT
fi
chmod 600 "$HDF5_OUT"/*.hdf5

echo "=== Label conversion START: $(date +%s) ==="
python scripts/moody_labels.py \
    --annot_dir "$ANNOT_DIR" \
    --target_dir "$LABELS_OUT" \
    --file_list "$ANNOT_FILELIST" \
    --num_workers 16
LABELS_EXIT=$?
echo "=== Label conversion END: $(date +%s) exit=$LABELS_EXIT ==="
if [ $LABELS_EXIT -ne 0 ]; then
    echo "FATAL: label conversion failed"
    exit $LABELS_EXIT
fi
chmod 600 "$LABELS_OUT"/*.csv

echo "=== Validation START: $(date +%s) ==="
python scripts/moody_validate.py \
    --hdf5_dir "$HDF5_OUT" \
    --labels_dir "$LABELS_OUT"
VALIDATE_EXIT=$?
echo "=== Validation END: $(date +%s) exit=$VALIDATE_EXIT ==="

rm -f "$PHYSIO_FILELIST" "$ANNOT_FILELIST"

echo "=== Final counts and sizes ==="
echo "HDF5 files: $(ls "$HDF5_OUT"/*.hdf5 | wc -l)"
echo "Label CSVs: $(ls "$LABELS_OUT"/*.csv | wc -l)"
du -sh "$HDF5_OUT" "$LABELS_OUT"

exit $VALIDATE_EXIT
