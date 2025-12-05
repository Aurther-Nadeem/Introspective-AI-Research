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
# 2. Run Experiments in Parallel (Control + Exp A)
echo "Running Experiments using Parallel Orchestrator..."
# This handles both Control and Exp A with 30 workers for max speed
python scripts/Remote/run_parallel_sweep.py --mode both --workers 30

# 3. Regrade Results in Parallel (BEFORE merging)
echo "Regrading results in parallel..."
# Only Exp A needs regrading (Injected Thoughts)
python scripts/Remote/run_parallel_sweep.py --mode regrade --workers 10

# 4. Merge Sharded Results
echo "Merging sharded results..."
# Since regrade overwrites in place, the files are now graded.
cat datasets/trials/meta_llama_llama_3.1_405b_instruct/sweep_injected_thoughts_L*.jsonl > "$RESULTS_FILE"
cat datasets/trials/meta_llama_llama_3.1_405b_instruct/sweep_control_bias_L*.jsonl > datasets/trials/meta_llama_llama_3.1_405b_instruct/sweep_control_bias.jsonl

# 5. Analyze Results
echo "Analyzing results..."
python scripts/Local/analyze_all.py "$OUTPUT_DIR"

echo "=== Evaluation Complete ==="
