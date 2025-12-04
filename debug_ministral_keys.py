import torch
from safetensors.torch import load_file
from pathlib import Path
from transformers.utils import cached_file

model_id = "mistralai/Ministral-3-8B-Reasoning-2512"

print(f"Inspecting keys for {model_id}...")

# Try to find the first safetensors file
try:
    # This might download if not cached, but we just downloaded it
    filename = cached_file(model_id, filename="model-00001-of-00004.safetensors")
    print(f"Loading keys from {filename}...")
    
    state_dict = load_file(filename)
    keys = list(state_dict.keys())
    
    print(f"\nTotal keys in shard: {len(keys)}")
    print("Sample keys:")
    for k in keys[:20]:
        print(f"  {k}")
        
    # Check for language_model prefix
    has_prefix = any(k.startswith("language_model.") for k in keys)
    print(f"\nHas 'language_model.' prefix: {has_prefix}")
    
except Exception as e:
    print(f"Error: {e}")
