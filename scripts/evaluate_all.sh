#!/bin/bash
# Evaluate all folds for an experiment and compute aggregate metrics
# Usage: bash scripts/evaluate_all.sh SleepEventLSTMClassifier_mesa_labels_BAS

EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: bash scripts/evaluate_all.sh <experiment_name>"
    echo "Example: bash scripts/evaluate_all.sh SleepEventLSTMClassifier_mesa_labels_BAS"
    exit 1
fi

echo "Evaluating experiment: $EXPERIMENT"
echo "================================================"

for fold in {0..9}; do
    echo "Evaluating fold $fold..."
    python sleepfm/pipeline/evaluate_sleep_staging.py \
        --output_path sleepfm/checkpoints/runs/${EXPERIMENT} \
        --split test \
        --fold $fold
done

echo ""
echo "================================================"
echo "Computing aggregate metrics..."
python scripts/compute_metrics.py \
    --checkpoint_dir sleepfm/checkpoints/runs/${EXPERIMENT}
