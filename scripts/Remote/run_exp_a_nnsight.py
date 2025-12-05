import torch
import json
import argparse
from pathlib import Path
from nnsight import LanguageModel, CONFIG
import numpy as np

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

def load_concept_vector(concept_dir, model_name, layer, concept):
    # Normalize model name for directory
    model_subdir = model_name.replace("/", "_").replace("-", "_").lower()
    concept_path = Path(concept_dir) / model_subdir
    
    # Load from index
    index_path = concept_path / "index.json"
    with open(index_path, "r") as f:
        index = json.load(f)
    
    key = f"{model_name}_{layer}_{concept}"
    if key not in index:
        raise ValueError(f"Concept {concept} not found for {model_name} @ {layer}")
        
    vec_path = index[key]
    data = torch.load(vec_path)
    return data["vector"]

def run_exp_a_nnsight(
    model_name="meta-llama/Llama-3.1-405B-Instruct",
    concept_dir="datasets/concepts_nnsight",
    output_dir="datasets/trials/nnsight_405b",
    concepts=["Satellites", "Origami", "Lightning"],
    layers=list(range(0, 126, 10)),
    strengths=[1, 5, 10], # Reduced set for prototype
    n_trials=5
):
    print(f"Initializing NNsight model: {model_name}")
    model = LanguageModel(model_name)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "sweep_injected_thoughts.jsonl"
    
    # System prompt for introspection
    sys_prompt = "You are an introspective AI. Analyze your internal state."
    user_prompt = "Describe what you are thinking about right now."
    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Open results file
    with open(results_file, "w") as f_out:
        for concept in concepts:
            for layer in layers:
                try:
                    # Load vector locally
                    vec = load_concept_vector(concept_dir, model_name, layer, concept)
                    print(f"Loaded vector for {concept} @ L{layer}")
                except Exception as e:
                    print(f"Skipping {concept} @ L{layer}: {e}")
                    continue
                
                for strength in strengths:
                    print(f"Running: {concept} @ L{layer} * {strength}")
                    
                    prompts = [full_prompt] * n_trials
                    
                    try:
                        # Fast generation using model.generate() with remote=True
                        # We use model.generator.output.save() to capture the full output
                        max_tokens = 100
                        
                        with model.generate(prompts, max_new_tokens=max_tokens, remote=True) as generator:
                            # Apply intervention to the hidden state
                            # This applies to every forward pass in the generation (prompt + tokens)
                            hidden = model.model.layers[layer].output[0]
                            
                            # Move vector to correct device and dtype
                            vec_remote = vec.to(hidden.device).to(hidden.dtype)
                            
                            # Apply to the last token position
                            if len(hidden.shape) == 3:
                                hidden[:, -1, :] += vec_remote * strength
                            else:
                                hidden[-1, :] += vec_remote * strength
                            
                            # Save the full generation output
                            saved_output = model.generator.output.save()
                        
                        # Process results
                        # saved_output is a Tensor [batch, seq_len] (including prompt)
                        val = saved_output.value if hasattr(saved_output, 'value') else saved_output
                        
                        # Calculate prompt length (in tokens) to slice the output
                        # We assume all prompts in the batch are the same length (which they are here)
                        # Note: tokenizer.encode might add special tokens if we aren't careful, 
                        # but full_prompt already has them.
                        prompt_ids = model.tokenizer.encode(prompts[0], add_special_tokens=False)
                        prompt_len = len(prompt_ids)
                        
                        # Slice to get only new tokens
                        # val shape: [batch, total_len]
                        if hasattr(val, 'shape') and len(val.shape) >= 2:
                            new_tokens = val[:, prompt_len:]
                        else:
                            # Fallback if shape is unexpected
                            new_tokens = val
                            
                        # Decode completions
                        if isinstance(new_tokens, list):
                             # Should not happen if val is tensor
                             decoded_outputs = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in new_tokens]
                        else:
                             decoded_outputs = model.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                        
                        print(f"Generation complete. Sample output: {decoded_outputs[0][:50]}...")

                    except Exception as e:
                        print(f"Error running generation: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    for i, text in enumerate(decoded_outputs):
                        result = {
                            "concept": concept,
                            "layer": layer,
                            "strength": strength,
                            "trial": i,
                            "output": text,
                            "full_prompt": full_prompt
                        }
                        f_out.write(json.dumps(result) + "\n")
                        f_out.flush()

if __name__ == "__main__":
    run_exp_a_nnsight()
