from nnsight import LanguageModel, CONFIG
import torch

CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

def debug_generate():
    model = LanguageModel("meta-llama/Llama-3.1-405B-Instruct")
    
    print("Starting generation...")
    with model.generate("Hello world", max_new_tokens=5, remote=True) as generator:
        print("Inside context")
        if hasattr(model, 'generator'):
            print("Saving model.generator.output")
            output = model.generator.output.save()
        else:
            print("model.generator not found")
            output = None
        
    print("Done.")
    print(f"Output type: {type(output)}")
    print(f"Output value: {output.value if hasattr(output, 'value') else output}")

if __name__ == "__main__":
    debug_generate()
