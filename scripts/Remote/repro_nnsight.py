
from nnsight import LanguageModel, CONFIG

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

print("Initializing model...")
model = LanguageModel("meta-llama/Llama-3.1-405B-Instruct")

print("Starting trace...")
with model.trace("hello", remote=True) as tracer:
    print("Inside trace block!")
    exec("h_exec = model.model.layers[0].output[0].save()")
    
print("Exited trace block.")
try:
    print(f"Captured h_exec: {h_exec}")
except NameError:
    print("h_exec not defined")
