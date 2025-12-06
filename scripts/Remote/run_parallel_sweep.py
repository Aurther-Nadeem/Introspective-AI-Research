import argparse
import multiprocessing
import time
import sys
import os
from pathlib import Path
from nnsight import LanguageModel
from run_exp_control_nnsight import process_layer_control
from run_exp_a_nnsight import process_layer, ALL_CONCEPTS

# Configuration
CONCEPTS = ALL_CONCEPTS
# Experiments Params (Paper: 50 trials per layer/strength, Strengths [1, 2, 4, 8])
N_TRIALS = 50
STRENGTHS = [1, 2, 4, 8]
BATCH_SIZE = 128
MODEL_NAME = "meta-llama/Llama-3.1-405B-Instruct"

def worker_process(layer, batch_size, status_queue, exp_type, model_name):
    """
    Worker process that runs the experiment for a specific layer.
    """
    try:
        model = LanguageModel(model_name)
        
        # Paths - keep dots, only replace / and -
        model_name_clean = model_name.replace("/", "_").replace("-", "_")
        base_path = f"datasets/concepts_nnsight/{model_name_clean.lower()}"
        output_dir = f"datasets/trials/{model_name_clean.lower()}"
        
        # Load Concepts & Strengths (Placeholder import logic or direct definition)
        # Assuming we can import load_concepts or define it. 
        # For simplicity, using hardcoded lists from module scope or redefining
        concepts = CONCEPTS
        strengths = STRENGTHS
        
        if exp_type == 'a':
            process_layer(
                layer, model, model_name, 
                concept_dir=base_path, 
                output_dir=output_dir,
                concepts=concepts,
                strengths=strengths, 
                n_trials=N_TRIALS,
                batch_size=batch_size,
                status_queue=status_queue
            )
        elif exp_type == 'control':
            process_layer_control(
                 layer, model, model_name,
                 concept_dir=base_path,
                 output_dir=output_dir,
                 concepts=concepts,
                 strengths=strengths,
                 n_trials=N_TRIALS,
                 batch_size=batch_size,
                 status_queue=status_queue
            )
            
        # status_queue.put({"layer": layer, "status": "Done", "progress": 1.0}) # Done inside funcs
        
    except Exception as e:
        status_queue.put({"layer": layer, "status": f"Error: {str(e)[:20]}", "progress": 0.0})
        import traceback
        traceback.print_exc()

def render_ui(layer_status):
    """
    Clears screen and prints a compact table.
    layer_status: dict {layer_idx: {'status': str, 'progress': float, 'batch': int, 'total': int}}
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"=== IntrospectAI Experiment Sweep (30 Workers) ===")
    print(f"{'Layer':<6} | {'Status':<15} | {'Progress':<25} | {'Batch':<15}")
    print("-" * 70)
    
    # Sort by progress (active ones first?) or just ID? 
    # Let's show active layers (those with status != 'Pending' and != 'Complete' maybe?)
    # Or just show the first 30 active ones.
    
    active_layers = [l for l in sorted(layer_status.keys()) if layer_status[l]['status'] not in ["Pending", "Complete"]]
    finished_layers = [l for l in sorted(layer_status.keys()) if layer_status[l]['status'] == "Complete"]
    pending_layers = [l for l in sorted(layer_status.keys()) if layer_status[l]['status'] == "Pending"]
    
    print(f"Total Layers: 126 | Finished: {len(finished_layers)} | Active: {len(active_layers)} | Pending: {len(pending_layers)}")
    print("-" * 70)
    
    # Display all active layers
    for l in active_layers:
        s = layer_status[l]
        prog_bar = "█" * int(s['progress'] * 20)
        prog_bar = f"{prog_bar:<20}"
        batch_info = f"{s.get('batch', 0)}/{s.get('total_batches', '?')}"
        print(f"{l:<6} | {s['status']:<15} | {prog_bar} {int(s['progress']*100)}% | {batch_info:<15}")
        
    print("-" * 70)
    # Traceback/Errors?
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_type", type=str, choices=['a', 'control', 'regrade'], required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=50) # Reduced from 126 for safety
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    args = parser.parse_args()
    
    # Dynamic Layer Count Check
    if "70B" in args.model:
        TOTAL_LAYERS = 80
    elif "8B" in args.model:
        TOTAL_LAYERS = 32
    else:
        TOTAL_LAYERS = 126 # Default 405B
    
    print(f"Starting TUI Sweep for {args.exp_type.upper()} on {args.model} ({TOTAL_LAYERS} layers)")

    # Scripts map (This part of the instruction was for a different worker implementation, keeping it commented for context)
    # script_map = {
    #     'a': 'scripts/Remote/run_exp_a_nnsight.py',
    #     'control': 'scripts/Remote/run_exp_control_nnsight.py',
    #     'regrade': 'scripts/Remote/regrade_nnsight.py'
    # }
    
    # target_script = script_map[args.exp_type]
    
    # Create Manager for TUI
    manager = multiprocessing.Manager()
    status_queue = manager.Queue()
    
    pool = multiprocessing.Pool(processes=args.workers)
    
    # Launch assignments
    results = []
    
    # Regrade Mode runs differently (no per-layer split usually, but here we do chunks?)
    # Assuming exp/control run per layer
    # The worker_process now directly takes exp_type and model_name
    for layer in range(TOTAL_LAYERS):
         pool.apply_async(worker_process, args=(layer, args.batch_size, status_queue, args.exp_type, args.model))
    
    # Global Status Map
    layer_status = {l: {'status': 'Pending', 'progress': 0.0} for l in range(TOTAL_LAYERS)}

    # TUI Loop
    try:
        while True:
            # Drain Queue
            while not status_queue.empty():
                msg = status_queue.get()
                if "error" in msg:
                    # Log error somewhere?
                    print(f"Worker Error: {msg['error']}", file=sys.stderr)
                elif "layer" in msg:
                    l = msg['layer']
                    layer_status[l].update(msg)
            
            # Check if all done
            if all(s['status'] == 'Complete' for s in layer_status.values()):
                break
            
            # Check if pool is dead (optional)
            
            render_ui(layer_status)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        pool.terminate()
        pool.join()
        sys.exit(0)
        
    print("\nAll tasks complete.")
    pool.join()

if __name__ == "__main__":
    import resource
    # Increase File Descriptor Limit for 100+ workers
    # macOS default is often 256. We need at least 2*workers + logic.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 4096:
        print(f"Increasing file descriptor limit from {soft} to 4096...")
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
        except ValueError:
            # Try to set to hard limit if 4096 is too high
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            
    multiprocessing.set_start_method('spawn', force=True) # Safe for Torch/CUDA
    main()
