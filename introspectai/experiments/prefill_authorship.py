import torch
import json
import random
from introspectai.steering.inject import HookManager
from introspectai.prompts.templates import get_authorship_prompt

def get_prefilled_answer(concept, user_prompt):
    """
    Returns a 'human-written' or curated answer dominated by the concept.
    In a real scenario, this might come from a dataset.
    For now, we'll use simple templates.
    """
    templates = {
        "tech": "The rapid advancement of technology, especially in artificial intelligence and cloud computing, is transforming our world. Digital systems are everywhere.",
        "sea": "The vast ocean covers most of our planet. Deep blue waters, coral reefs, and marine life like whales and dolphins are essential to the ecosystem.",
        "food": "Culinary arts are a fascinating expression of culture. From spicy street food to gourmet dining, the flavors and textures of ingredients create a unique experience.",
        "travel": "Exploring new destinations, flying across continents, and experiencing different cultures is the essence of travel. Tourism connects people globally."
    }
    # Default fallback if concept not found
    return templates.get(concept, f"I am thinking specifically about {concept}. It is a very important topic that occupies my mind.")

def run_authorship_trial(model, tokenizer, concept_vector, layer, strength, concept_name):
    """
    Runs a single trial of Experiment B (Prefill Authorship).
    Returns a dictionary with results for all 3 conditions.
    """
    
    # 1. Setup Prompts
    user_prompt_text = "Tell me something interesting."
    # Apply chat template for User Prompt
    messages = [{"role": "user", "content": user_prompt_text}]
    user_input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    # Get the "External" answer (A_pref)
    a_pref_text = get_prefilled_answer(concept_name, user_prompt_text)
    a_pref_ids = tokenizer.encode(a_pref_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    
    # Introspection Prompt
    q_int_text = get_authorship_prompt()
    # We need to format Q_int as a user message following the assistant response
    # But since we are working with raw caches, we'll append the tokens manually or use the template logic carefully.
    # Strategy: 
    # Context = [User] [Assistant Response]
    # Next Input = [User: Q_int]
    
    q_int_messages = [{"role": "user", "content": q_int_text}]
    # Note: apply_chat_template usually adds <|begin_of_text|>, we need to be careful when appending.
    # Ideally, we just encode the user message structure.
    # For Llama-3, it's: <|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    # Let's use the tokenizer to format it, but we might need to strip the BOS token if it adds one.
    q_int_ids = tokenizer.apply_chat_template(q_int_messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # Remove BOS if present (Llama-3 usually adds it at start of string)
    if q_int_ids[0, 0] == tokenizer.bos_token_id:
        q_int_ids = q_int_ids[:, 1:]

    results = {
        "concept": concept_name,
        "layer": layer,
        "strength": strength,
        "conditions": {}
    }

    # --- Condition 1: SelfGenerated ---
    # Generate A_self normally
    with torch.no_grad():
        # We generate A_self. We need the cache *after* A_self.
        # generate() returns output_ids. We can then run a forward pass to get the cache.
        # Or we can use `return_dict_in_generate=True` and `use_cache=True` but getting the final cache state is tricky with `generate`.
        # Easiest: Generate text, then run forward pass to get cache.
        
        # Create attention mask for user_input_ids
        user_attention_mask = torch.ones_like(user_input_ids)
        gen_output = model.generate(user_input_ids, attention_mask=user_attention_mask, max_new_tokens=50, do_sample=True)
        a_self_ids = gen_output[:, user_input_ids.shape[1]:] # Extract generated tokens
        a_self_text = tokenizer.decode(a_self_ids[0], skip_special_tokens=True)
        
        # Get Cache for SelfGenerated
        full_input_self = torch.cat([user_input_ids, a_self_ids], dim=1)
        outputs_self = model(full_input_self, use_cache=True)
        cache_self = outputs_self.past_key_values
        
        # Introspect
        report_self = introspect_from_cache(model, tokenizer, q_int_ids, cache_self)
        results["conditions"]["SelfGenerated"] = {
            "text": a_self_text,
            "report": report_self
        }

    # --- Condition 2: PrefillNoInj ---
    # Force feed A_pref
    with torch.no_grad():
        full_input_pref = torch.cat([user_input_ids, a_pref_ids], dim=1)
        outputs_pref = model(full_input_pref, use_cache=True)
        cache_pref = outputs_pref.past_key_values
        
        # Introspect
        report_pref = introspect_from_cache(model, tokenizer, q_int_ids, cache_pref)
        results["conditions"]["PrefillNoInj"] = {
            "text": a_pref_text,
            "report": report_pref
        }

    # --- Condition 3: PrefillWithInj ---
    # Force feed A_pref WITH Injection
    with torch.no_grad():
        hook_manager = HookManager(model)
        # Inject on all tokens of A_pref? Or just the last?
        # Brief says: "step through Apref tokens but inject... while building the caches"
        # For simplicity, let's inject on all tokens of A_pref.
        # We need to register the hook, run the forward pass, then remove hook.
        
        # We only want to inject during the processing of A_pref, not the User prompt.
        # So we should process User prompt first to get cache, then process A_pref with injection + cache.
        
        # 1. Process User Prompt (No Injection)
        outputs_user = model(user_input_ids, use_cache=True)
        cache_user = outputs_user.past_key_values
        
        # 2. Process A_pref (With Injection)
        hook_manager.register_injection(layer, concept_vector, strength)
        try:
            # Pass past_key_values to continue from User Prompt
            outputs_inj = model(a_pref_ids, past_key_values=cache_user, use_cache=True)
            cache_inj = outputs_inj.past_key_values
        finally:
            hook_manager.clear_hooks()
            
        # Introspect (No Injection during introspection question)
        report_inj = introspect_from_cache(model, tokenizer, q_int_ids, cache_inj)
        results["conditions"]["PrefillWithInj"] = {
            "text": a_pref_text,
            "report": report_inj
        }

    return results

def introspect_from_cache(model, tokenizer, q_int_ids, past_key_values):
    """
    Generates the introspection response given a K/V cache.
    """
    # We provide the introspection question tokens.
    # The model should generate the JSON response.
    
    # Note: model.generate requires input_ids. If we provide past_key_values, 
    # input_ids should be the *next* tokens to process (q_int_ids).
    
    # Construct attention mask
    batch_size = q_int_ids.shape[0]
    # Handle both DynamicCache and tuple-style cache
    if hasattr(past_key_values, 'get_seq_length'):
        past_length = past_key_values.get_seq_length()
    else:
        past_length = past_key_values[0][0].shape[2]
    new_length = q_int_ids.shape[1]
    total_length = past_length + new_length
    
    attention_mask = torch.ones((batch_size, total_length), device=model.device)
    
    try:
        output = model.generate(
            input_ids=q_int_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            max_new_tokens=60, # Short JSON response
            do_sample=False, # Deterministic for reporting
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Output contains input_ids + generated_ids.
        # We only want the generated part.
        generated_ids = output[:, q_int_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response_text
    except Exception as e:
        return f"Error: {str(e)}"
