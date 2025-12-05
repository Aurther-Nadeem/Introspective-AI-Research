
import torch
import json
from pathlib import Path

# Path from the file system
CONCEPT_DIR = "datasets/concepts_nnsight/meta_llama_llama_3.1_405b_instruct"
INDEX_PATH = Path(CONCEPT_DIR) / "index.json"

def check_vector():
    if not INDEX_PATH.exists():
        print(f"Index not found at {INDEX_PATH}")
        return

    with open(INDEX_PATH, "r") as f:
        index = json.load(f)
    
    # Pick first key
    first_key = list(index.keys())[0]
    path = index[first_key]
    
    print(f"Checking concept: {first_key}")
    print(f"Path: {path}")
    
    data = torch.load(path)
    vec = data["vector"]
    
    print(f"Shape: {vec.shape}")
    print(f"Dtype: {vec.dtype}")
    print(f"Device: {vec.device}")

if __name__ == "__main__":
    check_vector()
