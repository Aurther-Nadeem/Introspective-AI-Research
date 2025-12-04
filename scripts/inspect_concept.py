import torch
import fire
from introspectai.models.load import load_model
from introspectai.steering.concepts import ConceptStore

def main(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    concept_store="datasets/concepts",
    layer=16,
    concept="sea",
    top_k=20
):
    print(f"Inspecting concept '{concept}' at layer {layer} for {model_name}...")
    
    # Load model
    model, tokenizer = load_model(model_name)
    
    # Load vector
    store = ConceptStore(concept_store)
    try:
        vec = store.load(model_name, layer, concept) # [D]
    except KeyError:
        print(f"Concept not found.")
        return

    vec = vec.to(model.device).to(model.dtype)
    
    # Logit Lens: Apply the language modeling head to the vector
    # The LM head is usually model.lm_head
    # But first we might need to apply the final norm (RMSNorm/LayerNorm)
    # The residual stream at layer L usually goes through:
    # Remaining layers -> Final Norm -> LM Head
    # OR, if we just want to see what the vector *locally* represents, 
    # we can try projecting it directly via the embedding matrix (unembedding) 
    # or just passing it through the LM head (which includes unembedding).
    
    # For Llama 3:
    # model.model.norm is the final norm
    # model.lm_head is the output projection
    
    with torch.no_grad():
        # 1. Apply final norm (approximation of what happens at the end)
        # Note: This is an approximation because the vector is added at layer L, 
        # not at the very end. But it gives a sense of the semantic direction.
        normed_vec = model.model.norm(vec.unsqueeze(0)) # [1, D]
        
        # 2. Project to logits
        logits = model.lm_head(normed_vec) # [1, Vocab]
        
        # 3. Get top k
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        print(f"\nTop {top_k} tokens for concept vector:")
        for i in range(top_k):
            token_id = top_indices[0, i].item()
            token = tokenizer.decode([token_id])
            prob = top_probs[0, i].item()
            print(f"{i+1}. '{token}' ({prob:.4f})")

if __name__ == "__main__":
    fire.Fire(main)
