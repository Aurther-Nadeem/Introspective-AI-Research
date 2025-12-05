#!/bin/bash

# Set PYTHONPATH to include current directory for module imports
export PYTHONPATH=$PYTHONPATH:.

# Define paths
OUTPUT_DIR="datasets/trials/nnsight_405b"
RESULTS_FILE="$OUTPUT_DIR/sweep_injected_thoughts.jsonl"

echo "=== Starting Full NNsight Experiment Evaluation ==="

# 1. Clean up old results
if [ -f "$RESULTS_FILE" ]; then
    echo "Removing old results file: $RESULTS_FILE"
    rm "$RESULTS_FILE"
fi

# 2. Run Generation (Experiment A)
echo "Running Experiment A generation (this may take a while)..."
# 2. Run Generation (Experiment A)
echo "Running Experiment A generation (this may take a while)..."
python scripts/Remote/run_exp_a_nnsight.py

# 3. Regrade Results
echo "Regrading results..."
python scripts/Remote/regrade_nnsight.py "$RESULTS_FILE"

# 4. Run Analysis
echo "Running Analysis..."
python scripts/Local/analyze_all.py "$OUTPUT_DIR"

echo "=== Evaluation Complete ==="
