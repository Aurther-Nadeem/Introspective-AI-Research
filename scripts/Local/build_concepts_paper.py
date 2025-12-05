import fire
import torch
import random
from pathlib import Path
from tqdm import tqdm
from introspectai.models.load import load_model
from introspectai.steering.concepts import ConceptStore, whiten

def build_paper_concepts(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    concepts="sea,technology",
    random_words_file="datasets/random_words.txt",
    out_dir="datasets/concepts_paper",
    layers="20,22,24"
):
    """
    Builds concept vectors using the protocol from the paper:
    1. Collect activations for "Tell me about {word}."
    2. Subtract mean activations of "Tell me about {random_word}."
    3. Use the token immediately prior to Assistant's response.
    """
    if isinstance(concepts, str):
        concepts = concepts.split(",")
    if isinstance(layers, str):
        layers = [int(x) for x in layers.split(",")]
        
    print(f"Building paper-style concepts for: {concepts}")
    print(f"Layers: {layers}")
    
    # Load model
    model, tokenizer = load_model(model_name)
    
    # Prepare random words if not exist
    if not Path(random_words_file).exists():
        print("Generating random words...")
        # Simple list of common nouns
        words = ["apple", "chair", "history", "math", "river", "cloud", "idea", "person", "time", "way", 
                 "year", "work", "government", "day", "man", "world", "life", "part", "house", "course",
                 "case", "system", "place", "end", "group", "company", "party", "information", "school", "fact",
                 "money", "point", "example", "state", "business", "night", "area", "water", "thing", "family",
                 "head", "hand", "order", "john", "side", "home", "development", "week", "power", "country"]
        Path(random_words_file).write_text("\n".join(words))
        
    random_words = Path(random_words_file).read_text().strip().splitlines()
    
    # Helper to get activation for a word
    def get_activation(word, layer_idx):
        # "Tell me about {word}."
        # We need to apply chat template to get the exact prompt structure
        messages = [{"role": "user", "content": f"Tell me about {word}."}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
        
        # The token of interest is the LAST token of the input (just before generation starts)
        # This corresponds to the "token immediately prior to the Assistant's response"
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            
        # hidden_states is tuple of (layer_0, layer_1, ... layer_N)
        # layer_idx + 1 because index 0 is embeddings
        h = outputs.hidden_states[layer_idx + 1] # [1, T, D]
        return h[0, -1, :] # [D]

    # 1. Compute Mean of Random Words (Baseline)
    print("Computing baseline (random words)...")
    baselines = {l: [] for l in layers}
    
    for word in tqdm(random_words, desc="Baseline"):
        for layer in layers:
            act = get_activation(word, layer)
            baselines[layer].append(act)
            
    mean_baselines = {l: torch.stack(acts).mean(dim=0) for l, acts in baselines.items()}
    
    # 2. Compute Concept Vectors
    store = ConceptStore(out_dir)
    
    for concept in concepts:
        print(f"Processing concept: {concept}")
        for layer in layers:
            # Get activation for the concept word
            # The paper implies they might use multiple examples? 
            # "We collected the model’s activations in response to the prompt 'Tell me about {word}.'"
            # It sounds like just ONE prompt per concept word.
            # But to be robust, let's maybe do a few variations if 'concept' is a category?
            # The paper says "concept of a given word". So 1 word = 1 concept.
            
            concept_act = get_activation(concept, layer)
            
            # Vector = Concept - Mean_Baseline
            vec = concept_act - mean_baselines[layer]
            
            # Normalize
            vec = vec / (torch.norm(vec) + 1e-8)
            
            # Save
            stats = {
                "method": "paper_simple",
                "baseline_words": len(random_words),
                "layer": layer
            }
            store.save(model_name, layer, concept, vec, stats)
            
    print("Done building paper concepts.")

if __name__ == "__main__":
    fire.Fire(build_paper_concepts)
