import torch

def step_with_cache(model, input_ids, past_key_values, injector=None):
    """
    Performs a single decoding step or prefill using cached KV.
    """
    # If past_key_values is provided, we only feed the new tokens (input_ids)
    # input_ids should be [B, 1] for decoding or [B, T] for prefill
    
    with torch.no_grad():
        if injector:
            with injector:
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
        else:
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
    return outputs.logits, outputs.past_key_values
