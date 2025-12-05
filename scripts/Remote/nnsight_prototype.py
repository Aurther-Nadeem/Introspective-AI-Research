from nnsight import LanguageModel, CONFIG
import torch

CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

def run_nnsight_test():
    print("Initializing NNsight with Llama-3.1-405B-Instruct...")
    model = LanguageModel('meta-llama/Llama-3.1-405B-Instruct')

    prompt = "The Eiffel Tower is in the city of"
    print(f"Tracing prompt: '{prompt}'")

    # Remote execution
    with model.trace(prompt, remote=True):
        # 1. Access and Save a Hidden State (Read)
        # Layer 10, first token (just as a test)
        hidden_state = model.model.layers[10].output[0].save()
        
        # 2. Intervention (Write)
        # Let's try a dummy intervention: Zero out the output of Layer 11
        # Note: The exact path depends on the model architecture. 
        # For Llama, it's usually model.layers[i]...
        # We will just save output for now to verify connectivity.
        output = model.output.save()

    print("Execution complete.")
    print("Model Output:", output)
    print("Hidden State Shape:", hidden_state.shape)

if __name__ == "__main__":
    run_nnsight_test()
