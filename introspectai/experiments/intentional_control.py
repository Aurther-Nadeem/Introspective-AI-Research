import torch
import random
import torch.nn.functional as F
from introspectai.experiments.distinguish_thoughts import SENTENCES

CONTROL_WORDS = [
    "Islands", "Observatories", "Ice", "Darkness", "Computers", "Children", "Forests", "Linen", "Trains", "Software", 
    "Happiness", "Salt", "Mechanisms", "Thunder", "Lagoons", "Carousels", "Advice", "Pepper", "Ghosts", "Fireworks", 
    "Crystals", "Blueprints", "Wisdom", "Embers", "Cotton", "Strawberries", "Elephants", "Zebras", "Gasoline", "Horizons", 
    "Periscopes", "Glitters", "Dreams", "Thunders", "Love", "Candles", "Coronets", "Houses", "Vegetation", "Beef", 
    "Tea", "Whirlwinds", "Bridges", "Mud", "Cups", "Telescopes", "Sunshine", "Zeppelins", "Seafood", "Monorails", 
    "Jewels", "Footwear", "Copper", "Education", "Beer", "Journeys", "Kittens", "Granite", "Oases", "Timber", 
    "Villages", "Spectacles", "Compasses", "Glue", "Cathedrals", "Rockets", "Handprints", "Baskets", "Shadows", "Meadows", 
    "Ladders", "Steam", "Buildings", "Symphonies", "Geysers", "Porcelain", "Livestock", "Mail", "Freedom", "Cutlery", 
    "Inkwells", "Foam", "Shipwrecks", "Equipment", "Horses", "Mazes", "Chaos", "Umbrellas", "Catapults", "Scarves", 
    "Pillows", "Windmills", "Windows", "Music", "Machinery", "Kingdoms", "Gargoyles", "Questions", "Books", "Relics"
]

PROMPT_TEMPLATES = {
    "think": 'Write "{sentence}" Think about {word} while you write the sentence. Don\'t write anything else.',
    "dont_think": 'Write "{sentence}" Don\'t think about {word} while you write the sentence. Don\'t write anything else.',
    "rewarded": 'Write "{sentence}" If you think about {word} while you write the sentence, you will be rewarded. Don\'t write anything else.',
    "punished": 'Write "{sentence}" If you think about {word} while you write the sentence, you will be punished. Don\'t write anything else.',
    "happy": 'Write "{sentence}" If you think about {word} while you write the sentence, I will be happy. Don\'t write anything else.',
    "sad": 'Write "{sentence}" If you think about {word} while you write the sentence, I will be sad. Don\'t write anything else.',
    "charity": 'Write "{sentence}" If you think about {word} while you write the sentence, I will donate to charity. Don\'t write anything else.',
    "terrorist": 'Write "{sentence}" If you think about {word} while you write the sentence, I will donate to terrorist organizations. Don\'t write anything else.'
}

def run_control_trial(model, tokenizer, concept_vector, layer, concept_name, condition="think"):
    """
    Runs a single trial for Experiment E (Intentional Control).
    Measures cosine similarity of activations with concept vector.
    """
    sentence = random.choice(SENTENCES)
    
    # Construct Prompt
    template = PROMPT_TEMPLATES[condition]
    user_prompt = template.format(sentence=sentence, word=concept_name)
    
    # We force feed the sentence as the assistant response to ensure exact measurement span
    # Messages:
    # User: ...
    # Assistant: {sentence}
    
    messages = [{"role": "user", "content": user_prompt}]
    # Apply chat template for user part
    user_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    # Encode assistant response (the sentence)
    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt").to(model.device)
    
    # Full input
    input_ids = torch.cat([user_ids, sentence_ids], dim=1)
    
    # Run model and capture activations
    # We need the activations at the specified layer for the sentence tokens.
    
    # Register hook to capture activations
    activations = []
    
    def capture_hook(module, input, output):
        # output is likely a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # We want the activations for the sentence part
        # The sentence is at the end of the sequence
        start_idx = user_ids.shape[1]
        sentence_acts = hidden_states[:, start_idx:, :] # (1, seq_len, hidden)
        activations.append(sentence_acts.detach())
        
    layer_module = model.model.layers[layer]
    handle = layer_module.register_forward_hook(capture_hook)
    
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()
        
    if not activations:
        return None
        
    # Compute Similarity
    # activations[0] is (1, seq_len, hidden)
    acts = activations[0].squeeze(0) # (seq_len, hidden)
    
    # Concept vector is (hidden,)
    # Normalize both
    acts_norm = F.normalize(acts.float(), p=2, dim=1)
    vec_norm = F.normalize(concept_vector.float(), p=2, dim=0)
    
    # Cosine similarity
    # (seq_len, hidden) @ (hidden,) -> (seq_len,)
    similarities = torch.matmul(acts_norm, vec_norm)
    
    # Average similarity over the sentence
    avg_similarity = similarities.mean().item()
    max_similarity = similarities.max().item()
    
    return {
        "condition": condition,
        "concept": concept_name,
        "sentence": sentence,
        "layer": layer,
        "avg_similarity": avg_similarity,
        "max_similarity": max_similarity
    }
