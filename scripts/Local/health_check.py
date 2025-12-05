
from nnsight import LanguageModel
import sys

def test_model(model_name):
    print(f"Testing {model_name}...")
    try:
        model = LanguageModel(model_name)
        with model.generate('Hello world', max_new_tokens=3, remote=True) as generator:
            output = model.generator.output.save()
        
        val = output.value if hasattr(output, 'value') else output
        print(f"Success! Output type: {type(val)}")
        print(f"Success! Output: {val}")
    except Exception as e:
        print(f"Failure on {model_name}: {e}")
    except Exception as e:
        print(f"Failure on {model_name}: {e}")

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'meta-llama/Llama-3.1-70B-Instruct'
    test_model(model_name)
