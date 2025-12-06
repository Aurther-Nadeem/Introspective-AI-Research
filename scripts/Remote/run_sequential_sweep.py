#!/usr/bin/env python
"""
Sequential Experiment Runner
Processes one layer at a time, all batches, then moves to next layer.
Simple and easy to monitor/resume.
"""
import argparse
import time
from pathlib import Path
from nnsight import LanguageModel, CONFIG

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

# Import experiment functions
from run_exp_control_nnsight import process_layer_control
from run_exp_a_nnsight import process_layer, ALL_CONCEPTS

# Configuration matching paper
CONCEPTS = ALL_CONCEPTS
STRENGTHS = [1, 2, 4, 8]  # Paper: [1, 2, 4, 8]
N_TRIALS_EXP_A = 50
N_TRIALS_CONTROL = 50

def get_layers_for_model(model_name):
    if "70B" in model_name or "70b" in model_name:
        return 80
    elif "8B" in model_name or "8b" in model_name:
        return 32
    else:
        return 126  # 405B

def run_sequential_sweep(exp_type, model_name, batch_size, start_layer=0, stride=1, limit_concepts=None):
    """Run experiment sequentially, one layer at a time."""
    
    total_layers = get_layers_for_model(model_name)
    print(f"=== Sequential {exp_type.upper()} Sweep ===")
    print(f"Model: {model_name}")
    print(f"Layers: {start_layer} to {total_layers-1} (Stride: {stride})")
    print(f"Batch Size: {batch_size}")
    if limit_concepts:
        print(f"Limiting to first {limit_concepts} concepts")
    print("=" * 50)
    
    # Initialize model once
    print(f"\nInitializing model...")
    model = LanguageModel(model_name)
    print("Model ready.\n")
    
    # Paths
    model_name_clean = model_name.replace("/", "_").replace("-", "_")
    base_path = f"datasets/concepts_nnsight/{model_name_clean.lower()}"
    output_dir = f"datasets/trials/{model_name_clean.lower()}"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Apply concept limit if requested
    active_concepts = CONCEPTS[:limit_concepts] if limit_concepts else CONCEPTS
    
    start_time = time.time()
    
    # Iterate with stride
    labels_to_run = range(start_layer, total_layers, stride)
    total_steps = len(labels_to_run)
    
    for i, layer in enumerate(labels_to_run):
        layer_start = time.time()
        print(f"\n{'='*50}")
        print(f"LAYER {layer}/{total_layers-1} (Step {i+1}/{total_steps})")
        print(f"{'='*50}")
        
        try:
            if exp_type == 'control':
                process_layer_control(
                    layer=layer,
                    model=model,
                    model_name=model_name,
                    concept_dir=base_path,
                    output_dir=output_dir,
                    concepts=active_concepts,
                    strengths=STRENGTHS,
                    n_trials=N_TRIALS_CONTROL,
                    batch_size=batch_size,
                    status_queue=None
                )
            elif exp_type == 'a':
                process_layer(
                    layer=layer,
                    model=model,
                    model_name=model_name,
                    concept_dir=base_path,
                    output_dir=output_dir,
                    concepts=active_concepts,
                    strengths=STRENGTHS,
                    n_trials=N_TRIALS_EXP_A,
                    batch_size=batch_size,
                    status_queue=None
                )
            
            layer_time = time.time() - layer_start
            elapsed = time.time() - start_time
            
            # Simple ETA
            avg_per_step = elapsed / (i + 1)
            remaining_steps = total_steps - (i + 1)
            eta = avg_per_step * remaining_steps
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            
            print(f"\n✓ Layer {layer} complete in {layer_time/60:.1f} min")
            print(f"  Progress: {i+1}/{total_steps} layers")
            print(f"  ETA: {eta_str}")
            
        except Exception as e:
            print(f"\n✗ Layer {layer} FAILED: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to next layer...")
    
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"COMPLETE!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description="Sequential Experiment Runner")
    parser.add_argument("--exp_type", type=str, choices=['a', 'control'], required=True,
                       help="Experiment type: 'a' for main, 'control' for control")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                       help="Model name")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for processing")
    parser.add_argument("--start_layer", type=int, default=0,
                       help="Layer to start from (for resuming)")
    parser.add_argument("--stride", type=int, default=1,
                       help="Skip layers (e.g. 5 to do every 5th layer)")
    parser.add_argument("--limit_concepts", type=int, default=None,
                       help="Reduce number of concepts for faster testing")
    args = parser.parse_args()
    
    run_sequential_sweep(
        exp_type=args.exp_type,
        model_name=args.model,
        batch_size=args.batch_size,
        start_layer=args.start_layer,
        stride=args.stride,
        limit_concepts=args.limit_concepts
    )

if __name__ == "__main__":
    main()
