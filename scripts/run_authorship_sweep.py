import argparse
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from introspectai.steering.concepts import ConceptStore
from introspectai.experiments.prefill_authorship import run_authorship_trial
try:
    from transformers import AutoModelForCausal_LM, AutoTokenizer
except ImportError:
    try:
        from transformers.models.auto import AutoModelForCausal_LM, AutoTokenizer
    except ImportError:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM as AutoModelForCausal_LM
        from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast as AutoTokenizer

def load_model(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausal_LM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run Authorship Introspection Sweep (Experiment B)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--concepts", type=str, default="tech,sea")
    parser.add_argument("--layers", type=str, default="16,24")
    parser.add_argument("--strengths", type=str, default="1.0,2.0,4.0,8.0")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="datasets/trials")
    args = parser.parse_args()

    # Parse arguments
    concepts = args.concepts.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    strengths = [float(x) for x in args.strengths.split(",")]
    
    # Load Model
    model, tokenizer = load_model(args.model)
    
    # Load Concepts
    concept_store = ConceptStore("datasets/concepts")
    # Ensure vectors exist (assuming they were built by previous steps)
    
    output_file = Path(args.output_dir) / "sweep_authorship.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Authorship Sweep...")
    print(f"Concepts: {concepts}")
    print(f"Layers: {layers}")
    print(f"Strengths: {strengths}")
    print(f"Trials per condition: {args.n_trials}")
    
    total_trials = len(concepts) * len(layers) * len(strengths) * args.n_trials
    pbar = tqdm(total=total_trials)
    
    with open(output_file, 'a') as f:
        for concept_name in concepts:
            # Load vector
            try:
                vector = concept_store.load(args.model, layers[0], concept_name)
            except (KeyError, FileNotFoundError):
                vector = None
            if vector is None:
                print(f"Skipping {concept_name}: Vector not found.")
                pbar.update(len(layers) * len(strengths) * args.n_trials)
                continue
                
            for layer in layers:
                # Load specific layer vector
                try:
                    vector = concept_store.load(args.model, layer, concept_name)
                except (KeyError, FileNotFoundError):
                    vector = None
                if vector is None:
                    print(f"Skipping {concept_name} Layer {layer}: Vector not found.")
                    pbar.update(len(strengths) * args.n_trials)
                    continue
                    
                for strength in strengths:
                    for _ in range(args.n_trials):
                        try:
                            result = run_authorship_trial(
                                model, tokenizer, vector, layer, strength, concept_name
                            )
                            
                            # Add metadata
                            result["model"] = args.model
                            
                            f.write(json.dumps(result) + "\n")
                            f.flush()
                        except Exception as e:
                            print(f"Error in trial: {e}")
                        
                        pbar.update(1)
                        
    print(f"Sweep complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
