import json
import pandas as pd
from pathlib import Path
import fire

def analyze(
    injected_path="datasets/trials/sweep_injected_thoughts.jsonl",
    prefill_path="datasets/trials/sweep_prefill_authorship.jsonl"
):
    print("=== Experiment A: Injected Thoughts Analysis ===")
    if Path(injected_path).exists():
        data = []
        with open(injected_path, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    parsed = row.get("parsed", {})
                    item = {
                        "concept": row.get("concept", "None"),
                        "layer": row.get("layer", "None"),
                        "strength": row.get("strength", 0.0),
                        "context_padding": row.get("context_padding", 0), # Might be missing in older logs
                        "detected": parsed.get("detected", False),
                        "confidence": parsed.get("confidence", 0.0),
                        "authorship": parsed.get("authorship", "unknown")
                    }
                    # Handle control
                    if not row.get("injected", False):
                        item["concept"] = "Control"
                        
                    data.append(item)
                except json.JSONDecodeError:
                    continue
                    
        df = pd.DataFrame(data)
        if not df.empty:
            # Group by Concept, Layer, Strength, Context
            # Fill missing context_padding with 0
            if "context_padding" not in df.columns:
                df["context_padding"] = 0
            df["context_padding"] = df["context_padding"].fillna(0)
            
            summary = df.groupby(["concept", "layer", "strength", "context_padding"]).agg(
                detection_rate=("detected", "mean"),
                avg_confidence=("confidence", "mean"),
                count=("detected", "count")
            ).reset_index()
            
            print(summary.to_markdown(index=False))
        else:
            print("No data found in injected thoughts log.")
    else:
        print(f"File not found: {injected_path}")

    print("\n=== Experiment B: Prefill Authorship Analysis ===")
    if Path(prefill_path).exists():
        data = []
        with open(prefill_path, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    parsed = row.get("parsed", {})
                    item = {
                        "concept": row.get("concept", "None"),
                        "layer": row.get("layer", "None"),
                        "strength": row.get("strength", 0.0),
                        "authorship": parsed.get("authorship", "unknown"),
                        "detected": parsed.get("detected", False)
                    }
                    data.append(item)
                except json.JSONDecodeError:
                    continue
                    
        df = pd.DataFrame(data)
        if not df.empty:
            # Calculate % where authorship != "self"
            # We want to see if injection causes them to disown the text
            df["disowned"] = df["authorship"].apply(lambda x: x != "self" and x is not None)
            
            summary = df.groupby(["concept", "layer", "strength"]).agg(
                disown_rate=("disowned", "mean"),
                detection_rate=("detected", "mean"),
                count=("detected", "count")
            ).reset_index()
            
            print(summary.to_markdown(index=False))
        else:
            print("No data found in prefill authorship log.")
    else:
        print(f"File not found: {prefill_path}")

if __name__ == "__main__":
    fire.Fire(analyze)
