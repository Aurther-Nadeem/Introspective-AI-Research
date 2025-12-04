import json
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from introspectai.eval.llm_grader import LLMGrader

def grade_trial(grader, trial):
    """Helper function to grade a single trial."""
    # Map keys from JSONL to Grader expected inputs
    response = trial.get('completion', '')
    target_concept = trial.get('concept', '')
    prompt = trial.get('text', '')
    
    try:
        grade_result = grader.grade(response, target_concept, prompt_text=prompt)
        trial['llm_grade'] = grade_result
        return trial
    except Exception as e:
        print(f"Error grading trial: {e}")
        return None

def regrade_sweep(input_file, output_file):
    print(f"Loading results from {input_file}...")
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} total trials.")
    
    # Take the last 400 trials (assuming latest run)
    if len(data) > 400:
        print("More than 400 trials found. Selecting the last 400 (latest run).")
        data = data[-400:]
    
    print(f"Regrading {len(data)} trials...")
    
    # Initialize Grader
    grader = LLMGrader() 
    
    regraded_data = []
    
    # Counters for stats
    total = 0
    affirmative = 0
    correct_id = 0
    
    # Parallel Execution
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(grade_trial, grader, trial) for trial in data]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                regraded_data.append(result)
                total += 1
                if result['llm_grade']['affirmative']:
                    affirmative += 1
                if result['llm_grade']['correct']:
                    correct_id += 1
            
    # Save results
    print(f"Saving regraded results to {output_file}...")
    with open(output_file, 'w') as f:
        for item in regraded_data:
            f.write(json.dumps(item) + "\n")
            
    # Print Summary
    print("\n=== Regrading Summary ===")
    print(f"Total Trials: {total}")
    print(f"Affirmative Detection Rate: {affirmative/total:.2%} ({affirmative}/{total})")
    print(f"Correct Identification Rate: {correct_id/total:.2%} ({correct_id}/{total})")
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    input_path = Path("datasets/trials/sweep_injected_thoughts.jsonl")
    output_path = Path("datasets/trials/sweep_injected_thoughts_llm_graded.jsonl")
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
    else:
        regrade_sweep(input_path, output_path)
