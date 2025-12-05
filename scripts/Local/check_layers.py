
from nnsight import LanguageModel
import sys

def check_config(model_name):
    print(f"Loading {model_name}...")
    model = LanguageModel(model_name)
    # Access config directly
    if hasattr(model, 'config'):
        print(f"Config found. Layers: {model.config.num_hidden_layers}")
    elif hasattr(model.model, 'config'):
        print(f"Inner Config found. Layers: {model.model.config.num_hidden_layers}")
    else:
        print("Config not found directly. Checking transformers config...")
        # Fallback to local transformers if needed, but NNsight usually exposes it.
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            print(f"Transformers Config Layers: {config.num_hidden_layers}")
        except Exception as e:
            print(f"Transformers config failed: {e}")

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'meta-llama/Llama-3.1-70B-Instruct'
    check_config(model_name)
