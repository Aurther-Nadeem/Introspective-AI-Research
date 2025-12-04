import torch
import sys
print(f"Sys Path: {sys.path}")
print(f"Torch version: {torch.__version__}")

import transformers
print(f"Transformers version: {transformers.__version__}")
print(f"Transformers is_torch_available: {transformers.is_torch_available()}")

try:
    from transformers import LlamaForCausal_LM
    print("Success: LlamaForCausal_LM imported")
except ImportError as e:
    print(f"Error importing LlamaForCausal_LM: {e}")
