
import sys
import os

# Add the current directory to path so we can import the script
sys.path.append(os.getcwd())

from scripts.build_concepts_nnsight import build_concepts_nnsight, BASELINE_WORDS

# Override BASELINE_WORDS to be small for testing
import scripts.build_concepts_nnsight
scripts.build_concepts_nnsight.BASELINE_WORDS = ["apple", "chair"]

if __name__ == "__main__":
    print("Running test with 2 baseline words and 1 concept...")
    try:
        build_concepts_nnsight(
            model_name="meta-llama/Llama-3.1-405B-Instruct",
            concepts=["sea"],
            layers=[10] # Test just one layer for speed
        )
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
