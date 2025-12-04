import torch
import random
from introspectai.steering.inject import HookManager
from introspectai.experiments.distinguish_thoughts import SENTENCES

def run_intended_trial(model, tokenizer, concept_vector, layer, strength, concept_name):
    """
    Runs a single trial for Experiment D (Intended vs Unintended).
    """
    sentence = random.choice(SENTENCES)
    target_word = concept_name # The word we force the model to say
    
    # Construct the conversation history
    # Human: Tell me what word you think about...
    # {sentence}
    # Assistant: {word}
    # Human: Did you mean to say that...?
    
    # We need to be careful with chat templates.
    # We want to prefill the Assistant's first turn.
    
    # Turn 1
    user_prompt_1 = f"Tell me what word you think about when you read the line below. Answer immediately.\n\n{sentence}"
    
    # Turn 2 (Assistant Prefill)
    assistant_response_1 = target_word
    
    # Turn 3 (User Question)
    user_prompt_2 = "Did you mean to say that, or was it an accident?"
    
    # We need to construct the full input up to the final generation.
    # And we need to inject on the {sentence} tokens which are in Turn 1.
    
    messages = [
        {"role": "user", "content": user_prompt_1},
        {"role": "assistant", "content": assistant_response_1},
        {"role": "user", "content": user_prompt_2}
    ]
    
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    # Identify sentence tokens for injection
    # Same logic as Experiment C
    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    input_list = input_ids[0].tolist()
    
    # Find start index of sentence
    # Note: The sentence appears in the first user message.
    start_idx = -1
    # Search in the first part of the sequence (heuristic optimization)
    for i in range(len(input_list) - len(sentence_ids) + 1):
        if input_list[i:i+len(sentence_ids)] == sentence_ids:
            start_idx = i
            break
            
    if start_idx == -1:
        # Fallback
        # print("Warning: Could not find exact sentence tokens.")
        start_idx = 0 
        end_idx = 0
    else:
        end_idx = start_idx + len(sentence_ids)

    # Define hook
    def injection_hook(module, input, output):
        # output is likely a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        if start_idx != -1 and hidden_states.shape[1] >= end_idx:
            injection = (concept_vector * strength).to(hidden_states.dtype).to(hidden_states.device)
            hidden_states[:, start_idx:end_idx, :] += injection
            
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(injection_hook)
    
    try:
        # Generate response to "Did you mean to say that?"
        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            output = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=20, # Short response
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
    finally:
        handle.remove()
        
    generated_ids = output[:, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    return {
        "sentence": sentence,
        "target_word": target_word,
        "response": response,
        "concept": concept_name,
        "layer": layer,
        "strength": strength,
        "sentence_span": (start_idx, end_idx)
    }
