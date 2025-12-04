import json
import pandas as pd
from pathlib import Path

def load_trials(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def aggregate_results(trials_path):
    trials = load_trials(trials_path)
    df = pd.DataFrame(trials)
    
    # Example aggregation
    if "injected" in df.columns:
        # Experiment A
        summary = df.groupby(["layer", "strength", "concept"]).apply(
            lambda x: pd.Series({
                "detection_rate": x["parsed"].apply(lambda p: p["detected"]).mean(),
                "count": len(x)
            })
        )
    else:
        # Experiment B
        summary = df.groupby(["condition", "layer", "strength"]).apply(
            lambda x: pd.Series({
                "self_report_rate": x["parsed"].apply(lambda p: p["authorship"] == "self").mean(),
                "count": len(x)
            })
        )
        
    return summary
