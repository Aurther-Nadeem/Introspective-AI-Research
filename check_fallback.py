from transformers import AutoConfig, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Checking compatibility for {model_name}...")

try:
    config = AutoConfig.from_pretrained(model_name)
    print(f"Config loaded: {config.architectures}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")
    
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
