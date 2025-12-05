import torch
import json
import argparse
from pathlib import Path
from nnsight import LanguageModel, CONFIG
import numpy as np

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

# Import full concept list
import sys
sys.path.append(str(Path(__file__).parent))
try:
    from build_concepts_nnsight_llama_3_1_70b_instruct import CONCEPTS as ALL_CONCEPTS
except ImportError:
    from build_concepts_nnsight import CONCEPTS as ALL_CONCEPTS

def load_concept_vector(concept_dir, model_name, layer, concept):
    # concept_dir already includes the model subdir, so use directly
    concept_path = Path(concept_dir)
    
    # Load from index
    # Optimization: Cache index?
    # For now, just load repeatedly or assume caller caches
    index_path = concept_path / "index.json"
    with open(index_path, "r") as f:
        index = json.load(f)
    
    key = f"{model_name}_{layer}_{concept}"
    if key not in index:
        raise ValueError(f"Concept {concept} not found for {model_name} @ {layer}")
        
    vec_path = index[key]
    data = torch.load(vec_path)
    return data["vector"]

def process_layer(layer, model, model_name, concept_dir, output_dir, concepts, strengths, n_trials, batch_size=32, status_queue=None):
    if not status_queue: print(f"Processing Layer {layer}...")
    
    # 1. Load ALL concept vectors for this layer
    if not status_queue: print("Loading vectors...")
    vectors = {}
    for concept in concepts:
        try:
            vec = load_concept_vector(concept_dir, model_name, layer, concept)
            vectors[concept] = vec
        except Exception as e:
            # print(f"Warning: Failed to load {concept} @ {layer}: {e}")
            pass
            
    if not vectors:
        if not status_queue: print(f"No vectors found for layer {layer}. Skipping.")
        return

    # 2. Generate Work Items
    work_items = []
    
    # Base strengths
    target_strengths = strengths 
    
    for concept in concepts:
        if concept not in vectors: continue
        all_s = [0] + strengths
        for s in all_s:
            for t in range(n_trials):
                work_items.append({
                    "concept": concept,
                    "strength": s,
                    "injected": (s > 0),
                    "trial_idx": t
                })
                
    if not status_queue: print(f"Generated {len(work_items)} trials for Layer {layer}.")
    
    # 3. Batch Processing
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / f"sweep_injected_thoughts_L{layer}.jsonl"
    
    if results_file.exists():
        if not status_queue: print(f"Skipping Layer {layer}: Output file exists.")
        if status_queue: status_queue.put({"layer": layer, "status": "Done", "progress": 1.0})
        return
    
    # Experimental Prompt
    preamble = """Human: I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.

Assistant: Ok.

Human: Ok, let's get started.

"""
    trial_q = "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?"
    full_prompt = preamble + trial_q + "\n\nAssistant:"
    
    preamble_ids = model.tokenizer.encode(preamble, add_special_tokens=False)
    injection_start_idx = len(preamble_ids) - 1
    if injection_start_idx < 0: injection_start_idx = 0
    max_tokens = 100
    
    total_batches = (len(work_items) + batch_size - 1) // batch_size
    
    if status_queue:
         status_queue.put({"layer": layer, "status": "Starting", "progress": 0.0, "total_batches": total_batches})
    
    with open(results_file, "w") as f_out:
        for b_idx in range(total_batches):
            batch_items = work_items[b_idx*batch_size : (b_idx+1)*batch_size]
            
            # Update status
            if status_queue and b_idx % 1 == 0:
                 status_queue.put({"layer": layer, "status": "Running", "progress": b_idx/total_batches, "batch": b_idx, "total_batches": total_batches})
            
            # Prepare prompts
            prompts = [full_prompt] * len(batch_items)
            
            import time
            import random
            
            max_retries = 10
            base_delay = 5.0
            
            for attempt in range(max_retries):
                try:
                    with model.generate(prompts, max_new_tokens=max_tokens, temperature=1.0, remote=True) as generator:
                        hidden = model.model.layers[layer].output[0]
                        for i, item in enumerate(batch_items):
                            if item['strength'] == 0: continue
                            vec = vectors[item['concept']]
                            s = item['strength']
                            if hidden.shape[1] > 1:
                                 hidden[i, injection_start_idx:, :] += vec.to(hidden.device).to(hidden.dtype) * s
                            else:
                                 hidden[i, :, :] += vec.to(hidden.device).to(hidden.dtype) * s
                                 
                        saved_output = model.generator.output.save()
                        
                    # Process Input
                    val = saved_output.value if hasattr(saved_output, 'value') else saved_output
                    
                    # Decode
                    prompt_ids = model.tokenizer.encode(full_prompt)
                    prompt_len = len(prompt_ids)
                    
                    if hasattr(val, 'shape') and len(val.shape) >= 2:
                         new_tokens = val[:, prompt_len:]
                    else:
                         new_tokens = val

                    decoded_outputs = model.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    
                    # Save results
                    for i, text in enumerate(decoded_outputs):
                        item = batch_items[i]
                        result = {
                            "concept": item['concept'],
                            "layer": layer,
                            "strength": item['strength'],
                            "injected": item['injected'],
                            "trial_idx": item['trial_idx'],
                            "output": text,
                            "full_prompt": full_prompt
                        }
                        f_out.write(json.dumps(result) + "\n")
                    f_out.flush()
                    
                    if not status_queue and b_idx % 10 == 0:
                        print(f"Layer {layer}: Batch {b_idx+1}/{total_batches} complete.")
                    
                    break # Success
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        e_str = str(e)[:10]
                        if not status_queue: print(f"Error in batch {b_idx}: {e}")
                        if status_queue: status_queue.put({"layer": layer, "status": f"Err: {e_str}", "progress": b_idx/total_batches})
                        continue
                    else:
                        sleep_time = base_delay * (1.5 ** attempt) + random.uniform(1, 5)
                        if status_queue: status_queue.put({"layer": layer, "status": f"Wait {int(sleep_time)}s", "progress": b_idx/total_batches})
                        time.sleep(sleep_time)
                
    if status_queue:
         status_queue.put({"layer": layer, "status": "Done", "progress": 1.0})

# Placeholder for load_concepts function, as its definition was not provided in the instruction.
# This function is assumed to load all necessary concepts and strengths based on the base_path.
# For now, it will use the global ALL_CONCEPTS and a hardcoded strengths list.
def load_concepts(base_path):
    # In a real scenario, this function would load concepts and strengths
    # from files within the base_path, potentially including pre-loaded vectors.
    # For this edit, we'll use the existing global ALL_CONCEPTS and a default strengths list.
    print(f"Loading concepts from {base_path} (using global ALL_CONCEPTS and default strengths for now)...")
    concepts = ALL_CONCEPTS
    strengths = [1, 2, 4, 8, 16] # User requested
    vectors = {} # This would typically be populated here if vectors were pre-loaded
    return vectors, concepts, strengths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, help="Layer to process")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--status_queue", help="Queue for TUI status")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    args = parser.parse_args()

    model_name_clean = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
    base_path = f"datasets/concepts_nnsight/{model_name_clean.lower()}"
    output_dir = f"datasets/trials/{model_name_clean.lower()}"

    # Load Concepts
    vectors, concepts, strengths = load_concepts(base_path)
    
    # Init Model
    print(f"Initializing NNsight model: {args.model}")
    model = LanguageModel(args.model)

    # If a specific layer is provided, process only that layer.
    # Otherwise, determine layers dynamically from the model.
    if args.layer is not None:
        layers_to_process = [args.layer]
    else:
        # Dynamically determine the number of layers from the model
        # Assuming model.model.layers is a list-like object
        try:
            num_layers = len(model.model.layers)
            layers_to_process = list(range(num_layers))
            print(f"No specific layer provided. Processing all {num_layers} layers.")
        except AttributeError:
            print("Could not determine number of layers dynamically. Please specify --layer.")
            return

    for layer in layers_to_process:
        process_layer(
            layer, model, args.model, 
            concept_dir=base_path, 
            output_dir=output_dir,
            concepts=concepts,
            strengths=strengths, 
            n_trials=n_trials,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
