from nnsight import LanguageModel, CONFIG
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

def check_layers():
    print("Initializing model...")
    model = LanguageModel("meta-llama/Llama-3.1-405B-Instruct")
    print(f"Number of layers: {len(model.model.layers)}")

if __name__ == "__main__":
    check_layers()
