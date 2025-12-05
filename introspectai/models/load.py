import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def load_model(model_name_or_path, dtype="bfloat16", device_map="auto"):
    """
    Loads a HF decoder-only model and tokenizer.
    """
    logger.info(f"Loading model: {model_name_or_path} with dtype={dtype}")
    
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    
    if "Ministral" in model_name_or_path:
        print("Detected Ministral model, using mistral_common tokenizer wrapper...")
        from introspectai.models.mistral_wrapper import MistralCommonTokenizer
        tokenizer = MistralCommonTokenizer(is_tekken=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
    # MPS doesn't support bfloat16 well, fallback to float16
    if torch.backends.mps.is_available() and torch_dtype == torch.bfloat16:
        logger.warning("MPS detected, falling back to float16 as bfloat16 is not supported")
        torch_dtype = torch.float16
        
    if "Ministral" in model_name_or_path:
        from transformers import Ministral3ForCausalLM
        from safetensors.torch import load_file
        from transformers.utils import cached_file
        import glob
        import os
        
        print("Loading Ministral-3 with weight remapping...")
        
        # 1. Initialize model (random weights for now)
        # We use _fast_init=False to avoid expensive random init if possible, 
        # but from_pretrained might override. 
        # Actually, let's just instantiate it.
        model = Ministral3ForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        # 2. Load and remap weights
        # Find all safetensors files
        # We need to find the cached directory
        try:
            # Get the first file to find the directory
            first_file = cached_file(model_name_or_path, filename="model-00001-of-00004.safetensors")
            cache_dir = os.path.dirname(first_file)
            shard_files = sorted(glob.glob(os.path.join(cache_dir, "*.safetensors")))
            
            print(f"Found {len(shard_files)} shards in {cache_dir}")
            
            full_state_dict = {}
            for shard in shard_files:
                print(f"Loading shard {os.path.basename(shard)}...")
                shard_state = load_file(shard)
                for k, v in shard_state.items():
                    if k.startswith("language_model."):
                        new_k = k.replace("language_model.", "")
                        full_state_dict[new_k] = v
                    # Ignore vision_tower and multi_modal_projector
            
            print(f"Loading {len(full_state_dict)} remapped keys into model...")
            missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
            
        except Exception as e:
            print(f"Error remapping weights: {e}")
            print("WARNING: Model may be using random weights!")
            
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    model.eval()
    
    # Disable dropout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0
            
    return model, tokenizer

def get_layer_count(model):
    if hasattr(model, "config"):
        return model.config.num_hidden_layers
    # Fallback for some architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError("Could not determine layer count")

def get_layer(model, idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[idx]
    raise ValueError("Could not access layers")

def estimate_residual_rms(model, tokenizer, layer_idx, num_samples=100):
    """
    Estimates the RMS of the residual stream at a given layer.
    """
    # Use a dummy text for estimation if no dataset provided
    text = "The quick brown fox jumps over the lazy dog. " * 5
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    # hidden_states is a tuple of (layer_0, layer_1, ... layer_N)
    # layer_0 is embedding output. layer_i is output of block i-1.
    # So model.model.layers[i] input is hidden_states[i].
    # Output of layer i is hidden_states[i+1].
    
    # We want the RMS of the activation *at the injection site*.
    # If we inject at the start of the layer, we want the input to the layer.
    # If we inject at the end (residual), we want the output.
    # The prompt implies adding to the residual stream, usually meaning the state *between* layers.
    # Let's target the output of the specified layer index.
    
    # Note: HF hidden_states[0] is embeddings. hidden_states[1] is output of layer 0.
    # hidden_states[layer_idx + 1] is output of layer [layer_idx].
    
    if hasattr(outputs, "hidden_states"):
        h = outputs.hidden_states[layer_idx + 1] # [B, T, D]
        rms = torch.sqrt(torch.mean(h**2)).item()
        return rms
        
    return 1.0
