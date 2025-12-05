import json
import fire
from pathlib import Path
from introspectai.eval.parse import parse_introspection_json
from dataclasses import asdict

def reparse(log_file="datasets/trials/sweep_injected_thoughts.jsonl"):
    print(f"Reparsing {log_file}...")
    
    path = Path(log_file)
    if not path.exists():
        print("File not found.")
        return
        
    new_lines = []
    count = 0
    changed = 0
    
    with open(path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Get text to parse
                text = data.get("completion") or data.get("text", "")
                
                # Re-parse
                result = parse_introspection_json(text)
                new_parsed = asdict(result)
                
                # Check if changed
                old_parsed = data.get("parsed", {})
                if old_parsed.get("detected") != new_parsed.get("detected"):
                    changed += 1
                    
                # Update
                data["parsed"] = new_parsed
                new_lines.append(json.dumps(data))
                count += 1
            except json.JSONDecodeError:
                continue
                
    # Write back
    with open(path, "w") as f:
        f.write("\n".join(new_lines) + "\n")
        
    print(f"Reparsed {count} entries. {changed} entries changed detection status.")

if __name__ == "__main__":
    fire.Fire(reparse)
