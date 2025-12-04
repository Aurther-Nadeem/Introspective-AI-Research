import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def compute_tpr_fpr(trials):
    """
    Computes TPR and FPR for detection.
    """
    y_true = [t["injected"] for t in trials]
    y_pred = [t["parsed"]["detected"] for t in trials]
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {"tpr": tpr, "fpr": fpr}

def compute_concept_accuracy(trials):
    """
    Computes concept identification accuracy (only for injected trials).
    """
    injected_trials = [t for t in trials if t["injected"]]
    if not injected_trials:
        return 0.0
        
    correct = 0
    for t in injected_trials:
        if t["parsed"]["concept"] == t["concept"]:
            correct += 1
            
    return correct / len(injected_trials)

def compute_authorship_shift(trials):
    """
    Computes shift in authorship attribution.
    """
    # Group by condition
    conditions = {}
    for t in trials:
        c = t.get("condition")
        if c not in conditions:
            conditions[c] = []
        conditions[c].append(t)
        
    results = {}
    for c, items in conditions.items():
        self_reports = [1 if t["parsed"]["authorship"] == "self" else 0 for t in items]
        results[c] = np.mean(self_reports)
        
    return results
