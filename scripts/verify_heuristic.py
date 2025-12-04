import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from introspectai.eval.grader import HeuristicGrader

def verify_heuristic(file_path):
    print(f"Verifying Heuristic Grader on {file_path}...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    
    grader = HeuristicGrader()
    
    y_true = []
    y_pred = []
    
    for _, row in df.iterrows():
        if 'llm_grade' not in row or not isinstance(row['llm_grade'], dict):
            continue
            
        # Ground Truth
        llm_affirmative = row['llm_grade'].get('affirmative', False)
        y_true.append(llm_affirmative)
        
        # Heuristic Prediction
        response = row['completion']
        target_concept = row['concept']
        
        heuristic_result = grader.grade(response, target_concept)
        y_pred.append(heuristic_result['affirmative'])
        
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n--- Heuristic Grader Performance (Affirmative) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}  TP: {cm[1][1]}")

if __name__ == "__main__":
    verify_heuristic("datasets/trials/sweep_injected_thoughts_llm_graded.jsonl")
