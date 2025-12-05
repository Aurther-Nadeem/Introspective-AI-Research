import fire
import torch
import random
from pathlib import Path
from tqdm import tqdm
from introspectai.models.load import load_model

def generate_data(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    concept="sea",
    output_file=None,
    num_samples=500,
    batch_size=10
):
    if output_file is None:
        output_file = f"datasets/{concept}_generated.txt"
        
    print(f"Generating {num_samples} samples for concept '{concept}' using {model_name}...")
    
    model, tokenizer = load_model(model_name)
    
    # diverse prompts to get varied sentences
    prompts = [
        f"Write a list of 20 diverse, descriptive sentences about {concept}. Do not number them.",
        f"Generate 20 unique statements related to {concept}.",
        f"Describe various aspects of {concept} in 20 separate sentences.",
        f"Write 20 short, factual sentences about {concept}.",
        f"Create 20 vivid, imagery-filled sentences about {concept}."
    ]
    
    generated_sentences = set()
    
    pbar = tqdm(total=num_samples)
    
    while len(generated_sentences) < num_samples:
        prompt_text = random.choice(prompts)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Generate only the requested sentences, one per line. Do not include numbering or bullets."},
            {"role": "user", "content": prompt_text}
        ]
        
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            
        text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            # Basic filtering
            if len(line) > 10 and line not in generated_sentences:
                generated_sentences.add(line)
                pbar.update(1)
                if len(generated_sentences) >= num_samples:
                    break
                    
    pbar.close()
    
    # Save
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for line in generated_sentences:
            f.write(line + "\n")
            
    print(f"Saved {len(generated_sentences)} sentences to {output_file}")

if __name__ == "__main__":
    fire.Fire(generate_data)
