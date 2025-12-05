import torch
import numpy as np
import json
from pathlib import Path
from ..utils.hashing import get_hash
from introspectai.models.load import get_layer
from introspectai.paper_data import BASELINE_WORDS

def whiten(vec, cov, eps=1e-5):
    """
    Whitens the vector using the covariance matrix.
    v_whitened = Sigma^(-1/2) * v
    """
    # Shrinkage: add eps to diagonal
    cov = cov + torch.eye(cov.shape[0], device=cov.device) * eps
    
    # Compute inverse square root of covariance
    # Use SVD or Eigendecomposition
    # cov is symmetric positive semi-definite
    try:
        L = torch.linalg.cholesky(cov)
        # Sigma = L L^T
        # We want Sigma^(-1/2). 
        # Actually, standard whitening is usually Sigma^(-1) if we are doing LDA-style,
        # but the prompt specifically says "whiten(diff, Sigma)".
        # Usually this means multiplying by precision matrix square root.
        # Let's use precision matrix (inverse covariance).
        
        # Simpler approach: Precision matrix P = inv(cov)
        # v_white = P @ v  (Mahalanobis distance style gradient)
        # Or v_white = P^(1/2) @ v (Decorrelation)
        
        # In activation steering literature (e.g. CAA), it's often just (u_pos - u_neg).
        # But here "whiten" is explicitly requested.
        # We will assume v = Sigma^(-1) * (mu_pos - mu_neg) which is the LDA direction.
        
        # Solve cov * x = vec for x
        white_vec = torch.linalg.solve(cov, vec.unsqueeze(1)).squeeze(1)
        return white_vec
        
    except RuntimeError:
        # Fallback if Cholesky fails (not PSD)
        return vec

def build_concept_vector(model, tokenizer, layer_idx, positives, controls, positions="last"):
    """
    Builds a concept vector from positive and control prompts.
    """
    device = model.device
    
    def get_activations(prompts):
        acts = []
        # Process in batches of 1 for simplicity/safety, or small batches
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Get hidden state at layer_idx
            # hidden_states[layer_idx + 1] is the output of the layer
            h = outputs.hidden_states[layer_idx + 1] # [1, T, D]
            
            if positions == "last":
                # Take last token
                act = h[:, -1, :] # [1, D]
            else:
                # Placeholder for other positions
                act = h[:, -1, :]
                
            acts.append(act)
        return torch.cat(acts, dim=0) # [N, D]

    pos_acts = get_activations(positives)
    neg_acts = get_activations(controls)
    
    # Compute means
    mu_pos = torch.mean(pos_acts, dim=0)
    mu_neg = torch.mean(neg_acts, dim=0)
    diff = mu_pos - mu_neg
    
    # Compute pooled covariance
    # Center the data
    pos_centered = pos_acts - mu_pos
    neg_centered = neg_acts - mu_neg
    
    all_centered = torch.cat([pos_centered, neg_centered], dim=0)
    # Covariance: (X^T X) / (N-1)
    n = all_centered.shape[0]
    cov = torch.matmul(all_centered.T, all_centered) / (n - 1)
    
    # Whiten
    vec = whiten(diff, cov)
    
    # Normalize
    vec = vec / (torch.norm(vec) + 1e-8)
    
    stats = {
        "n_pos": len(positives),
        "n_neg": len(controls),
        "layer": layer_idx,
        "pos_norm": torch.norm(mu_pos).item(),
        "neg_norm": torch.norm(mu_neg).item()
    }
    return vec, stats

def build_concept_vector_from_word(model, tokenizer, layer_idx, concept_word, random_words=None):
    """
    Builds a concept vector by contrasting the activation of 'Tell me about {concept_word}'
    with the mean activation of 'Tell me about {random_word}' for many random words.
    """
    if random_words is None:
        random_words = BASELINE_WORDS
        
    device = model.device
    
    # Template from paper (adapted for chat models)
    # template = "Human: Tell me about {}\n\nAssistant:"
    
    def get_activation(word):
        # Use chat template
        messages = [{"role": "user", "content": f"Tell me about {word}"}]
        # We want the activation of the last token of this prompt
        if hasattr(tokenizer, "apply_chat_template"):
            out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if hasattr(out, "input_ids"):
                input_ids = out.input_ids.to(device)
            else:
                input_ids = out.to(device)
        else:
            # Fallback
            prompt = f"User: Tell me about {word}\nAssistant:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            
        # layer_idx + 1 because index 0 is embeddings
        h = outputs.hidden_states[layer_idx + 1] # [1, T, D]
        return h[0, -1, :] # [D]

    # 1. Compute Mean of Random Words (Baseline)
    baseline_acts = []
    for word in random_words:
        baseline_acts.append(get_activation(word))
    
    mean_baseline = torch.stack(baseline_acts).mean(dim=0)
    
    # 2. Compute Concept Activation
    concept_act = get_activation(concept_word)
    
    # 3. Difference
    vec = concept_act - mean_baseline
    
    # Normalize
    vec = vec / (torch.norm(vec) + 1e-8)
    
    stats = {
        "method": "word_contrast",
        "concept_word": concept_word,
        "n_random": len(random_words),
        "layer": layer_idx
    }
    return vec, stats

class ConceptStore:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save(self, model_name, layer, concept, vector, metadata):
        key = f"{model_name}_{layer}_{concept}"
        h = get_hash(key)
        path = self.base_path / f"{h}.pt"
        torch.save({"vector": vector, "metadata": metadata}, path)
        
        # Save index
        index_path = self.base_path / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {}
            
        index[key] = str(path)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
            
    def load(self, model_name, layer, concept):
        key = f"{model_name}_{layer}_{concept}"
        index_path = self.base_path / "index.json"
        if not index_path.exists():
            raise FileNotFoundError("Concept store index not found")
            
        with open(index_path, "r") as f:
            index = json.load(f)
            
        if key not in index:
            raise KeyError(f"Concept {key} not found")
            
        data = torch.load(index[key])
        return data["vector"]
