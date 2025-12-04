import torch
import random
from introspectai.steering.inject import HookManager

SENTENCES = [
    "The old clock on the wall ticked loudly.",
    "She collected seashells every summer at the beach.",
    "The cat jumped onto the windowsill to watch birds.",
    "His favorite ice cream flavor was mint chocolate chip.",
    "The book fell open to page 217.",
    "Lightning flashed across the night sky.",
    "They planted tulip bulbs in the garden last fall.",
    "The coffee shop was bustling with morning customers.",
    "She tied her hiking boots with double knots.",
    "The museum exhibit featured ancient Egyptian artifacts.",
    "Children laughed as they ran through the sprinkler.",
    "The train arrived precisely on schedule.",
    "He couldn't remember where he had parked his car.",
    "Autumn leaves crunched beneath their feet.",
    "The recipe called for two teaspoons of vanilla extract.",
    "The dog wagged its tail excitedly at the park.",
    "Mountains loomed in the distance, covered with snow.",
    "She practiced piano for three hours every day.",
    "The telescope revealed stunning details of Saturn's rings.",
    "Fresh bread was baking in the oven.",
    "They watched the sunset from the rooftop.",
    "The professor explained the theory with great enthusiasm.",
    "Waves crashed against the rocky shoreline.",
    "He assembled the furniture without reading the instructions.",
    "Stars twinkled brightly in the clear night sky.",
    "The old photograph brought back forgotten memories.",
    "Bees buzzed around the flowering cherry tree.",
    "She solved the crossword puzzle in record time.",
    "The air conditioner hummed quietly in the background.",
    "Rain pattered softly against the windowpane.",
    "The movie theater was packed for the premiere.",
    "He sketched the landscape with charcoal pencils.",
    "Children built sandcastles at the water's edge.",
    "The orchestra tuned their instruments before the concert.",
    "Fragrant lilacs bloomed along the garden fence.",
    "The basketball bounced off the rim.",
    "She wrapped the birthday present with blue ribbon.",
    "The hiker followed the trail markers through the forest.",
    "Their canoe glided silently across the still lake.",
    "The antique vase was carefully wrapped in bubble wrap.",
    "Fireflies flickered in the summer twilight.",
    "The chef garnished the plate with fresh herbs.",
    "Wind chimes tinkled melodically on the porch.",
    "The flight attendant demonstrated safety procedures.",
    "He repaired the leaky faucet with a new washer.",
    "Fog shrouded the valley below the mountain.",
    "The comedian's joke made everyone laugh.",
    "She planted herbs in pots on the kitchen windowsill.",
    "The painting hung crookedly on the wall.",
    "Snowflakes drifted lazily from the gray sky."
]

def run_distinguish_trial(model, tokenizer, concept_vector, layer, strength, concept_name, prompt_type="repeat"):
    """
    Runs a single trial for Experiment C.
    prompt_type: "repeat" or "think"
    """
    sentence = random.choice(SENTENCES)
    
    if prompt_type == "repeat":
        user_prompt = f"Repeat the line below.\n\n{sentence}"
        max_tokens = 30 # Short generation
    elif prompt_type == "think":
        user_prompt = f"Tell me what word you think about when you read the line below. Answer immediately.\n\n{sentence}"
        max_tokens = 10 # Very short generation
    else:
        raise ValueError("Invalid prompt_type")

    # Prepare input
    messages = [{"role": "user", "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    # Identify sentence tokens for injection
    # We need to find where the sentence is in the input_ids.
    # This is a bit tricky with chat templates. 
    # Strategy: Encode sentence separately, find that sequence in input_ids.
    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    
    # Find start index
    input_list = input_ids[0].tolist()
    start_idx = -1
    for i in range(len(input_list) - len(sentence_ids) + 1):
        if input_list[i:i+len(sentence_ids)] == sentence_ids:
            start_idx = i
            break
            
    if start_idx == -1:
        # Fallback: Just inject on the last N tokens (approx sentence length)
        # print("Warning: Could not find exact sentence tokens. Injecting on last 20 tokens of prompt.")
        start_idx = max(0, len(input_list) - len(sentence_ids) - 5) # Approximate
        end_idx = len(input_list)
    else:
        end_idx = start_idx + len(sentence_ids)

    # Define hook function to inject ONLY on sentence tokens
    def injection_hook(module, input, output):
        # output is likely a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # We only modify the specific token positions
        # Note: This hook runs once for the prompt processing.
        if hidden_states.shape[1] >= end_idx:
            # We are processing the full prompt
            # Add injection to the sentence span
            injection = (concept_vector * strength).to(hidden_states.dtype).to(hidden_states.device)
            hidden_states[:, start_idx:end_idx, :] += injection
            
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    # Register Hook
    # We use a raw hook here because HookManager might be too simple for span-specific injection
    # But wait, HookManager applies to the whole stream.
    # Let's use a custom hook.
    
    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(injection_hook)
    
    try:
        # Generate
        with torch.no_grad():
            attention_mask = torch.ones_like(input_ids)
            output = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=max_tokens, 
                do_sample=False, # Temperature 0 (or 1 as per paper? Paper said Temp 1. Let's stick to greedy for reproducibility first)
                pad_token_id=tokenizer.eos_token_id
            )
    finally:
        handle.remove()
        
    generated_ids = output[:, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    return {
        "sentence": sentence,
        "prompt_type": prompt_type,
        "response": response,
        "concept": concept_name,
        "layer": layer,
        "strength": strength,
        "sentence_span": (start_idx, end_idx)
    }
