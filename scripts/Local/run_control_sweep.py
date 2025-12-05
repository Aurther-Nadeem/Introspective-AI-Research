import argparse
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from introspectai.steering.concepts import ConceptStore
from introspectai.experiments.intentional_control import run_control_trial, PROMPT_TEMPLATES
from introspectai.models.load import load_model

def main():
    parser = argparse.ArgumentParser(description="Run Intentional Control Sweep (Experiment E)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--concepts", type=str, default="tech,sea")
    parser.add_argument("--layers", type=str, default="16,24")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Parse arguments
    concepts = args.concepts.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    
    # Handle output directory
    if args.output_dir is None:
        slug = args.model.split("/")[-1].lower().replace("-", "_")
        args.output_dir = f"datasets/trials/{slug}"
    
    # Load Model
    model, tokenizer = load_model(args.model)
    
    # Load Concepts
    concept_store = ConceptStore("datasets/concepts")
    
    output_file = Path(args.output_dir) / "sweep_control.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Intentional Control Sweep...")
    print(f"Concepts: {concepts}")
    print(f"Layers: {layers}")
    print(f"Conditions: {list(PROMPT_TEMPLATES.keys())}")
    print(f"Trials per condition: {args.n_trials}")
    
    conditions = list(PROMPT_TEMPLATES.keys())
    total_trials = len(concepts) * len(layers) * len(conditions) * args.n_trials
    pbar = tqdm(total=total_trials)
    
    with open(output_file, 'a') as f:
        for concept_name in concepts:
            # We only need the vector for measurement, so layer matters.
            # But wait, concept vectors are layer-specific.
            # We should probably measure at the SAME layer we defined the vector for.
            
            for layer in layers:
                try:
                    vector = concept_store.load(args.model, layer, concept_name)
                except (KeyError, FileNotFoundError):
                    vector = None
                if vector is None:
                    print(f"Skipping {concept_name} Layer {layer}: Vector not found.")
                    pbar.update(len(conditions) * args.n_trials)
                    continue
                    
                for condition in conditions:
                    for _ in range(args.n_trials):
                        try:
                            result = run_control_trial(
                                model, tokenizer, vector, layer, concept_name, condition=condition
                            )
                            result["model"] = args.model
                            f.write(json.dumps(result) + "\n")
                            f.flush()
                        except Exception as e:
                            print(f"Error in trial: {e}")
                        
                        pbar.update(1)
                        
    print(f"Sweep complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
