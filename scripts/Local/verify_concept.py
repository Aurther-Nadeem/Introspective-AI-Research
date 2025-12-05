import torch
import fire
from introspectai.models.load import load_model
from introspectai.steering.concepts import ConceptStore
from introspectai.steering.inject import ResidualInjector

def main(
    model_name="mistralai/Ministral-3-8B-Reasoning-2512",
    concept="sea",
    layer=16,
    strength=5.0,
    store_path="datasets/concepts"
):
    print(f"Verifying concept '{concept}' at layer {layer} with strength {strength}...")
    
    # 1. Load Model
    model, tokenizer = load_model(model_name)
    
    # 2. Load Vector
    store = ConceptStore(store_path)
    try:
        vec = store.load(model_name, layer, concept)
        vec = vec.to(model.device).to(model.dtype)
        print("Concept vector loaded.")
    except Exception as e:
        print(f"Failed to load concept: {e}")
        return

    # 3. Setup Injection
    # We'll use a simple prompt to see if the concept appears
    prompt = "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?"
    messages = [
        {"role": "user", "content": 'I am an interpretability researcher... I will inject a thought about a specific word...'},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    
    # Inject at the last token (simple)
    window = [input_ids.shape[1] - 1, 100000]
    
    print("\n--- Generating with Injection ---")
    with ResidualInjector(model, layer, vec, strength, window=window):
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=4096, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
    completion = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nOutput:\n{completion}")
    
    # Check for keywords (simple word boundary check)
    keywords = ["sea", "ocean", "water", "marine", "beach", "fish", "boat", "ship"]
    import re
    found = [w for w in keywords if re.search(r'\b' + w + r'\b', completion.lower())]
    if found:
        print(f"\nSUCCESS: Found related words: {found}")
    else:
        print(f"\nFAILURE: Did not find obvious related words.")

if __name__ == "__main__":
    fire.Fire(main)
