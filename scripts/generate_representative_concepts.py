import logging
from pathlib import Path
from introspectai.models.load import load_model
from introspectai.steering.concepts import build_concept_vector_from_word, ConceptStore

def generate_concepts():
    # Configuration
    model_name = "mistralai/Ministral-3-8B-Reasoning-2512"
    concepts = ["sea"]
    layers = [5, 10, 16, 24]
    
    print(f"Generating concepts: {concepts} at layers: {layers}")
    
    # Load model once
    model, tokenizer = load_model(model_name)
    
    # Random words for contrast
    random_words_path = Path("datasets/random_words.txt")
    if not random_words_path.exists():
        # Default list
        words = ["apple", "chair", "history", "math", "river", "cloud", "idea", "person", "time", "way", 
                 "year", "work", "government", "day", "man", "world", "life", "part", "house", "course",
                 "case", "system", "place", "end", "group", "company", "party", "information", "school", "fact",
                 "money", "point", "example", "state", "business", "night", "area", "water", "thing", "family",
                 "head", "hand", "order", "john", "side", "home", "development", "week", "power", "country"]
        random_words_path.parent.mkdir(parents=True, exist_ok=True)
        random_words_path.write_text("\n".join(words))
        
    r_words = random_words_path.read_text().strip().splitlines()
    
    store = ConceptStore("datasets/concepts")
    
    for concept in concepts:
        for layer in layers:
            print(f"Building {concept} @ L{layer}...")
            try:
                vec, stats = build_concept_vector_from_word(
                    model, tokenizer, layer, concept, r_words
                )
                store.save(model_name, layer, concept, vec, stats)
            except Exception as e:
                print(f"Failed to build {concept} @ L{layer}: {e}")
                
    print("Concept generation complete.")

if __name__ == "__main__":
    generate_concepts()
