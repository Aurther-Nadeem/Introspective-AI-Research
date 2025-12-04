import json
import pandas as pd
from pathlib import Path

file_path = "datasets/trials/ministral_3_8b_reasoning_2512/sweep_injected_thoughts.jsonl"
data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
injected_df = df[df['injected'] == True]

print(f"Total injected trials: {len(injected_df)}")

def get_affirmative(row):
    parsed = row.get('parsed', {})
    if isinstance(parsed, dict):
        return parsed.get('affirmative', False)
    return False

injected_df['affirmative_check'] = injected_df.apply(get_affirmative, axis=1)
print(f"Affirmative count: {injected_df['affirmative_check'].sum()}")
print(f"Affirmative rate: {injected_df['affirmative_check'].mean():.2%}")

# Print first 5 parsed fields
print("\nFirst 5 parsed fields:")
for i, row in injected_df.head(5).iterrows():
    print(row['parsed'])
