import fire
import logging
import torch
from introspectai.models.load import load_model
from introspectai.steering.concepts import build_concept_vector, ConceptStore
from introspectai.utils.logging import setup_logging

def main(
    model: str = "gpt2",
    out_dir: str = "datasets/concepts",
    layer: int = 5,
    concept: str = "test",
    positives: str = "positives.txt", # Path to file
    controls: str = "controls.txt"    # Path to file
    method: str = "word",
    random_words: str = "datasets/random_words.txt"
):
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print(f"Building concept '{concept}' at layer {layer} for {model}...")
    
    model_obj, tokenizer = load_model(model)
    
    if method == "word":
        print("Using paper methodology (word contrast)...")
        if not Path(random_words).exists():
            # Generate default random words if missing
            words = ["apple", "chair", "history", "math", "river", "cloud", "idea", "person", "time", "way", 
                     "year", "work", "government", "day", "man", "world", "life", "part", "house", "course",
                     "case", "system", "place", "end", "group", "company", "party", "information", "school", "fact",
                     "money", "point", "example", "state", "business", "night", "area", "water", "thing", "family",
                     "head", "hand", "order", "john", "side", "home", "development", "week", "power", "country"]
            Path(random_words).parent.mkdir(parents=True, exist_ok=True)
            Path(random_words).write_text("\n".join(words))
            
        r_words = Path(random_words).read_text().strip().splitlines()
        
        # We need to import build_concept_vector_from_word
        from introspectai.steering.concepts import build_concept_vector_from_word
        
        vec, stats = build_concept_vector_from_word(
            model_obj, tokenizer, layer, concept, r_words
        )
    else:
        print("Using dataset methodology...")
        pos_prompts = Path(positives).read_text().strip().splitlines()
        neg_prompts = Path(controls).read_text().strip().splitlines()
        
        vec, stats = build_concept_vector(
            model_obj, tokenizer, layer, pos_prompts, neg_prompts
        )
    
    store = ConceptStore(out_dir)
    store.save(model, layer, concept, vec, stats)
    print(f"Saved concept vector to {out_dir}")

if __name__ == "__main__":
    fire.Fire(main)
