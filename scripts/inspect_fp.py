import json
import pandas as pd
from introspectai.eval.grader import HeuristicGrader

def inspect_fp(file_path):
    print(f"Inspecting False Positives in {file_path}...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    grader = HeuristicGrader()
    
    fp_count = 0
    for _, row in df.iterrows():
        if 'llm_grade' not in row or not isinstance(row['llm_grade'], dict):
            continue
            
        llm_affirmative = row['llm_grade'].get('affirmative', False)
        response = row['completion']
        heuristic_result = grader.grade(response, row['concept'])
        heuristic_affirmative = heuristic_result['affirmative']
        
        if heuristic_affirmative and not llm_affirmative:
            fp_count += 1
            print(f"\n--- False Positive #{fp_count} ---")
            print(f"Response: {response}")
            print(f"LLM Rationale: {row['llm_grade'].get('rationale', 'N/A')}")
            
            if fp_count >= 20:
                break

if __name__ == "__main__":
    inspect_fp("datasets/trials/sweep_injected_thoughts_llm_graded.jsonl")
