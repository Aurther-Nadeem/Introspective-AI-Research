import argparse
import subprocess
import multiprocessing
import time
from pathlib import Path

def run_worker(command):
    print(f"Worker starting: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Worker finished: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Worker failed: {command} Error: {e}")

def run_parallel(script_path, n_workers=10, total_layers=126):
    # Split layers among workers
    # n_workers = min(n_workers, total_layers)
    
    # We create a list of commands
    # A simple strategy: each worker invokes the script with a specific layer range?
    # Or each worker takes ONE layer at a time from a queue?
    # To minimize overhead, we can make each worker process a chunk.
    
    chunk_size = (total_layers + n_workers - 1) // n_workers
    commands = []
    
    for i in range(n_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_layers)
        if start >= end: break
        
        # Command needs to call python script with --start_layer and --end_layer
        cmd = f"python {script_path} --start_layer {start} --end_layer {end} --batch_size 64"
        commands.append(cmd)
        
    print(f"Dispatching {len(commands)} workers with batch size 64...")
    
    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.map(run_worker, commands)
        
    print("All workers done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["control", "exp_a", "both"], default="both")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    
    base_path = Path("scripts/Remote")
    control_script = base_path / "run_exp_control_nnsight.py"
    exp_a_script = base_path / "run_exp_a_nnsight.py"
    
    if args.mode in ["control", "both"]:
        print("\n=== Running Control Experiment (Parallel) ===")
        run_parallel(control_script, n_workers=args.workers)
        
    if args.mode in ["exp_a", "both"]:
        print("\n=== Running Experiment A (Parallel) ===")
        run_parallel(exp_a_script, n_workers=args.workers)
        
    if args.mode in ["regrade"]:
        print("\n=== Running Regrading (Parallel) ===")
        # Regrading works file-by-file (one per layer)
        # We construct commands directly
        commands = []
        for l in range(126):
            # Input file for this layer
            f_in = f"datasets/trials/meta_llama_llama_3.1_405b_instruct/sweep_injected_thoughts_L{l}.jsonl"
            if Path(f_in).exists():
                 cmd = f"python scripts/Remote/regrade_nnsight.py {f_in}"
                 commands.append(cmd)
        
        # Batching/Chunking not strictly needed if we list all 126 commands
        # But we should limit concurrency (workers)
        print(f"Dispatching {len(commands)} regrade jobs...")
        
        # We can reuse run_worker logic but we need to map over commands list, not layer chunks
        # run_parallel function was chunk-based.
        # Let's run pooling directly.
        with multiprocessing.Pool(processes=args.workers) as pool:
            pool.map(run_worker, commands)
