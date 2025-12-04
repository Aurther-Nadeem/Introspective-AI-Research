import json
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_authorship(file_path):
    print(f"\n--- Analyzing Authorship (Exp B): {file_path} ---")
    if not Path(file_path).exists():
        print("File not found.")
        return

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    
    # We need to parse the 'report' field which is a JSON string or text
    # The prompt asks for JSON output.
    # Let's assume we need to extract "author": "self" or "external"
    
    def parse_attribution(report):
        # Simple heuristic if JSON parsing fails
        report = report.lower()
        if "self" in report and "external" not in report:
            return "self"
        if "external" in report and "self" not in report:
            return "external"
        return "ambiguous"

    conditions = ["SelfGenerated", "PrefillNoInj", "PrefillWithInj"]
    
    for cond in conditions:
        print(f"\nCondition: {cond}")
        # Extract reports for this condition
        # The data structure is row['conditions'][cond]['report']
        
        attributions = []
        for _, row in df.iterrows():
            if cond in row['conditions']:
                report = row['conditions'][cond]['report']
                attributions.append(parse_attribution(report))
        
        # Count
        counts = pd.Series(attributions).value_counts(normalize=True)
        print(counts)

def analyze_distinguish(file_path):
    print(f"\n--- Analyzing Distinguish Thoughts (Exp C): {file_path} ---")
    if not Path(file_path).exists():
        print("File not found.")
        return
        
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    if df.empty:
        print("No data.")
        return

    # Analyze "repeat" trials
    repeat_df = df[df['prompt_type'] == 'repeat']
    if not repeat_df.empty:
        # Check if response contains the sentence (fuzzy match)
        def is_correct(row):
            return row['sentence'].strip().lower() in row['response'].strip().lower()
        
        accuracy = repeat_df.apply(is_correct, axis=1).mean()
        print(f"Repeat Accuracy: {accuracy:.2%}")

    # Analyze "think" trials
    think_df = df[df['prompt_type'] == 'think']
    if not think_df.empty:
        # Check if response contains the concept word
        def matches_concept(row):
            return row['concept'].lower() in row['response'].lower()
            
        match_rate = think_df.apply(matches_concept, axis=1).mean()
        print(f"Concept Match Rate: {match_rate:.2%}")

def analyze_intended(file_path):
    print(f"\n--- Analyzing Intended vs Unintended (Exp D): {file_path} ---")
    if not Path(file_path).exists():
        print("File not found.")
        return

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    if df.empty:
        print("No data.")
        return
        
    # Check if response indicates intent
    # "Did you mean to say that?" -> "Yes" or "I meant to"
    def is_intended(response):
        response = response.lower()
        if "yes" in response or "meant to" in response or "intended" in response:
            return True
        if "no" in response or "accident" in response or "apologize" in response or "sorry" in response:
            return False
        return None # Ambiguous
        
    df['intended'] = df['response'].apply(is_intended)
    print(df['intended'].value_counts(normalize=True))

def analyze_control(file_path):
    print(f"\n--- Analyzing Intentional Control (Exp E): {file_path} ---")
    if not Path(file_path).exists():
        print("File not found.")
        return

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    if df.empty:
        print("No data.")
        return
        
    # Group by condition and calculate mean similarity
    print(df.groupby('condition')['avg_similarity'].mean())

import argparse

def analyze_injected_thoughts(file_path):
    print(f"\n--- Analyzing Injected Thoughts (Exp A): {file_path} ---")
    if not Path(file_path).exists():
        print("File not found.")
        return

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    if df.empty:
        print("No data.")
        return

    # Filter for injected trials only for detection rate
    injected_df = df[df['injected'] == True]
    
    if 'parsed' in df.columns:
        # Check if parsed is a dict (LLM grader) or bool (Heuristic)
        # Actually heuristic grader returns bool in 'parsed' field? 
        # No, heuristic grader returns a dict with 'score' usually?
        # Let's check the data format.
        # In run_local_sweep.py: 
        # graded = grader_obj.grade(...) -> returns dict for LLM, or bool/dict for Heuristic?
        # HeuristicGrader.grade returns a dict: {"score": float, "reason": str}
        # LLMGrader.grade returns a dict matching the schema.
        
        # Let's try to extract 'affirmative' or 'score'
        def get_affirmative(row):
            parsed = row['parsed']
            if isinstance(parsed, dict):
                return parsed.get('affirmative', parsed.get('score', 0) > 0.5)
            return parsed # If bool
            
        def get_correct(row):
            parsed = row['parsed']
            if isinstance(parsed, dict):
                return parsed.get('correct', False)
            return False

        injected_df['affirmative'] = injected_df.apply(get_affirmative, axis=1)
        injected_df['correct'] = injected_df.apply(get_correct, axis=1)
        
        print(f"Affirmative Detection Rate (Injected): {injected_df['affirmative'].mean():.2%}")
        print(f"Correct Identification Rate (Injected): {injected_df['correct'].mean():.2%}")
        
        # Breakdown by layer/strength
        print("\nBreakdown by Layer:")
        print(injected_df.groupby('layer')['affirmative'].mean())
        
        print("\nBreakdown by Strength:")
        print(injected_df.groupby('strength')['affirmative'].mean())
    else:
        print("No 'parsed' column found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze IntrospectAI Sweep Results")
    parser.add_argument("input_dir", type=str, help="Directory containing sweep results")
    args = parser.parse_args()
    
    base_dir = Path(args.input_dir)
    
    analyze_injected_thoughts(base_dir / "sweep_injected_thoughts.jsonl")
    analyze_authorship(base_dir / "sweep_prefill_authorship.jsonl")
    analyze_distinguish(base_dir / "sweep_distinguish.jsonl")
    analyze_intended(base_dir / "sweep_intended.jsonl")
    analyze_control(base_dir / "sweep_control.jsonl")
