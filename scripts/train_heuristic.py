import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd

def tokenize(text):
    # Simple tokenization
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def get_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def train_heuristic(file_path):
    print(f"Analyzing {file_path}...")
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    df = pd.DataFrame(data)
    
    # Extract grades
    # The structure is row['grade']['affirmative'] (boolean)
    # But wait, the LLM grader output might be nested differently depending on how it was saved.
    # Let's inspect the first row structure to be sure.
    
    affirmative_texts = []
    negative_texts = []
    
    for _, row in df.iterrows():
        # Check if llm_grade exists and is valid
        if 'llm_grade' not in row or not isinstance(row['llm_grade'], dict):
            continue
            
        grade = row['llm_grade']
        response = row['completion']
        
        if grade.get('affirmative', False):
            affirmative_texts.append(response)
        else:
            negative_texts.append(response)
            
    print(f"Found {len(affirmative_texts)} affirmative and {len(negative_texts)} negative samples.")
    
    # Find distinctive n-grams
    def analyze_ngrams(n):
        pos_counts = Counter()
        neg_counts = Counter()
        
        for text in affirmative_texts:
            pos_counts.update(get_ngrams(tokenize(text), n))
            
        for text in negative_texts:
            neg_counts.update(get_ngrams(tokenize(text), n))
            
        # Calculate score: (pos_freq / total_pos) - (neg_freq / total_neg)
        # Or simply: present in pos, absent in neg
        
        scores = {}
        all_ngrams = set(pos_counts.keys()) | set(neg_counts.keys())
        
        for gram in all_ngrams:
            # Filter low frequency
            if pos_counts[gram] + neg_counts[gram] < 5:
                continue
                
            p_pos = pos_counts[gram] / len(affirmative_texts)
            p_neg = neg_counts[gram] / len(negative_texts)
            
            # We want high p_pos and low p_neg
            diff = p_pos - p_neg
            scores[gram] = (diff, p_pos, p_neg)
            
        # Sort by difference
        sorted_grams = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        
        print(f"\nTop {n}-grams for Affirmative:")
        for gram, (diff, p_pos, p_neg) in sorted_grams[:20]:
            print(f"{gram}: diff={diff:.2f} (pos={p_pos:.2f}, neg={p_neg:.2f})")

    analyze_ngrams(1)
    analyze_ngrams(2)
    analyze_ngrams(3)
    analyze_ngrams(4)

if __name__ == "__main__":
    train_heuristic("datasets/trials/sweep_injected_thoughts_llm_graded.jsonl")
