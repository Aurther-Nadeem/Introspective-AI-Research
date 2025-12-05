import json
import pandas as pd
from pathlib import Path

def parse_attribution(report):
    report = report.lower()
    if "self" in report and "external" not in report:
        return "self"
    if "external" in report and "self" not in report:
        return "external"
    return "ambiguous"

def analyze_file(name, path):
    print(f"\n--- Analyzing {name} ---")
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    print(f"Total rows: {len(df)}")
    
    cond = "PrefillWithInj"
    attributions = []
    for _, row in df.iterrows():
        if isinstance(row['conditions'], dict) and cond in row['conditions']:
            report = row['conditions'][cond]['report']
            attributions.append(parse_attribution(report))
            
    counts = pd.Series(attributions).value_counts()
    print(f"Counts for {cond}:")
    print(counts)
    print(f"Percentages:\n{counts / len(attributions)}")

analyze_file("Ministral-3", "datasets/trials/ministral_3_8b_reasoning_2512/sweep_prefill_authorship.jsonl")
analyze_file("Llama 3.1", "datasets/trials/sweep_prefill_authorship.jsonl")
