import json
import fire
from pathlib import Path

def inspect(
    log_file="datasets/trials/sweep_injected_thoughts.jsonl",
    concept="tech",
    layer=16,
    strength=2.0
):
    print(f"Inspecting {log_file} for Concept={concept}, Layer={layer}, Strength={strength}")
    
    found = 0
    with open(log_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Check if this entry matches criteria
                # Note: keys might differ slightly depending on how Config was serialized
                # The log entry usually has 'config' dict and 'result' dict
                
                # Flat structure
                if (data.get("concept") == concept and 
                    data.get("layer") == layer and 
                    data.get("strength") == strength):
                    
                    print("\n" + "="*80)
                    print(f"Trial ID: {data.get('trial_id')}")
                    print(f"Context Padding: {data.get('context_padding')}")
                    print("-" * 40)
                    print("RAW COMPLETION:")
                    # The 'text' field contains the full text. 
                    # We might want to try to separate prompt/completion if possible, 
                    # but for now just printing the whole thing or the end is fine.
                    # The 'completion' field might not exist if it wasn't saved separately.
                    # Let's check if 'completion' exists, otherwise print 'text'
                    
                    content = data.get("completion")
                    if not content:
                        content = data.get("text", "")[-500:] # Last 500 chars if full text
                        print("(Showing last 500 chars of 'text')")
                    
                    print(content.strip())
                    print("-" * 40)
                    print(f"Parsed Detection: {data.get('parsed', {}).get('detected')}")
                    print("="*80 + "\n")
                    found += 1
                    
            except json.JSONDecodeError:
                continue
                
    print(f"Found {found} matching entries.")

if __name__ == "__main__":
    fire.Fire(inspect)
