import json
import argparse
from pathlib import Path
from tqdm import tqdm
from introspectai.eval.grader import HeuristicGrader
# from introspectai.eval.llm_grader import LLMGrader # Optional

def regrade_nnsight(input_file, output_file):
    print(f"Loading results from {input_file}...")
    if not Path(input_file).exists():
        print("Input file not found.")
        return

    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} trials.")
    
    # Initialize Grader
    # Default to Heuristic for speed/cost, can make configurable
    grader = HeuristicGrader()
    
    regraded_data = []
    
    for trial in tqdm(data, desc="Grading"):
        # Map NNsight output to Grader input
        response = trial.get('output', '')
        concept = trial.get('concept', '')
        prompt = trial.get('full_prompt', '')
        
        # Grade
        try:
            # Heuristic grader usually takes (response, target_concept)
            # Check signature: grade(self, response, target_concept, prompt_text=None)
            graded = grader.grade(response, concept, prompt_text=prompt)
            
            # Add fields required by analyze_all.py
            trial['parsed'] = graded
            trial['injected'] = True # NNsight script currently only runs injected trials
            
            # Ensure other fields exist
            if 'completion' not in trial:
                trial['completion'] = response
            
            regraded_data.append(trial)
        except Exception as e:
            print(f"Error grading trial: {e}")
            
    # Save results
    print(f"Saving graded results to {output_file}...")
    with open(output_file, 'w') as f:
        for item in regraded_data:
            f.write(json.dumps(item) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("--output_file", help="Path to output JSONL file (default: overwrite input)")
    args = parser.parse_args()
    
    output = args.output_file if args.output_file else args.input_file
    regrade_nnsight(args.input_file, output)
