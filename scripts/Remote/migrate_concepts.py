import json
import shutil
from pathlib import Path

def migrate():
    base_dir = Path("datasets/concepts_nnsight")
    model_name = "meta_llama_3.1_405b_instruct"
    target_dir = base_dir / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = base_dir / "index.json"
    if not index_path.exists():
        print("No index.json found!")
        return

    with open(index_path, "r") as f:
        index = json.load(f)
        
    new_index = {}
    
    print(f"Migrating {len(index)} entries...")
    
    for key, old_path_str in index.items():
        old_path = Path(old_path_str)
        filename = old_path.name
        
        # Check if file exists in old location
        if not old_path.exists():
            # Maybe it was already moved? check target
            new_path = target_dir / filename
            if new_path.exists():
                print(f"File {filename} already moved.")
            else:
                print(f"WARNING: File {filename} not found in {old_path} or {new_path}")
                continue
        else:
            # Move file
            new_path = target_dir / filename
            shutil.move(str(old_path), str(new_path))
            
        # Update index with new path (relative to CWD)
        # The original code used str(path) where path was relative to CWD.
        # So we want "datasets/concepts_nnsight/meta_llama_3.1_405b_instruct/xyz.pt"
        new_index[key] = str(new_path)
        
    # Save new index in the target directory
    new_index_path = target_dir / "index.json"
    with open(new_index_path, "w") as f:
        json.dump(new_index, f, indent=2)
        
    print(f"Migration complete. New index saved to {new_index_path}")
    
    # Optionally remove old index if empty?
    # Or keep it but empty?
    # I'll rename it to index.json.bak
    index_path.rename(base_dir / "index.json.bak")
    print("Renamed old index to index.json.bak")

if __name__ == "__main__":
    migrate()
