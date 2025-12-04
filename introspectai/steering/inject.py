import torch
from dataclasses import dataclass
from typing import Union, List, Optional

@dataclass
class InjectionSpec:
    concept_id: str
    layer_idx: int
    strength: float
    schedule: str = "last_token"
    window: Optional[Union[List[int], str]] = None
    seed: int = 42

class ResidualInjector:
    def __init__(self, model, layer_idx, vector, strength, schedule="last_token", window="last_token"):
        self.model = model
        self.idx = layer_idx
        self.vector = vector.to(model.device)
        self.strength = strength
        self.schedule = schedule
        self.window = window
        self._orig = None
        self._hook_handle = None

    def __enter__(self):
        # Locate the layer
        # Support multiple architectures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama, Mistral
            layer = self.model.model.layers[self.idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT2, DistilGPT2
            layer = self.model.transformer.h[self.idx]
        else:
            raise ValueError("Unsupported model architecture for injection")
            
        self._orig = layer.forward
        
        def wrapped_forward(hidden_states, *args, **kwargs):
            # hidden_states: [B, T, D]
            B, T, D = hidden_states.shape
            
            # Create mask based on window/schedule
            mask = torch.zeros(B, T, 1, device=hidden_states.device, dtype=hidden_states.dtype)
            
            if self.window == "last_token":
                mask[:, -1, :] = 1.0
            elif isinstance(self.window, list) or isinstance(self.window, tuple):
                t0, t1 = self.window
                # Ensure indices are valid
                t0 = max(0, t0)
                t1 = min(T, t1)
                if t1 > t0:
                    mask[:, t0:t1, :] = 1.0
            elif self.window == "all":
                 mask[:, :, :] = 1.0
            
            # Apply injection
            # vector is [D], reshape to [1, 1, D]
            delta = self.strength * self.vector.view(1, 1, -1) * mask
            
            hidden_states = hidden_states + delta
            
            return self._orig(hidden_states, *args, **kwargs)
            
        # Monkey patch
        # Note: This is a simplified monkey patch. 
        # For robust implementation, we might want to use register_forward_hook 
        # but that operates on output. To modify input, we need register_forward_pre_hook 
        # or monkey patch the forward method itself to modify hidden_states in-place or before passing to orig.
        # The user spec asked for monkey patching forward.
        
        # However, to properly bind the method we need to be careful.
        # A safer way for this specific request (modifying residual stream) 
        # is to wrap the forward method.
        
        layer.forward = wrapped_forward
        self._hook_handle = layer # Keep ref to restore
        
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._hook_handle:
            self._hook_handle.forward = self._orig

class HookManager:
    def __init__(self, model):
        self.model = model
        self.injectors = []

    def register_injection(self, layer_idx, vector, strength, window="all"):
        # Create and enter the injector context
        injector = ResidualInjector(self.model, layer_idx, vector, strength, window=window)
        injector.__enter__()
        self.injectors.append(injector)

    def clear_hooks(self):
        # Exit all injectors
        for injector in self.injectors:
            injector.__exit__(None, None, None)
        self.injectors = []
