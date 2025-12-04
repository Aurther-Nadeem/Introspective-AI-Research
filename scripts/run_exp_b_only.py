#!/usr/bin/env python
"""Run only Experiment B (Prefill Authorship) using existing concept vectors."""
import torch
import fire
from pathlib import Path
from introspectai.models.load import load_model, get_layer_count
from introspectai.steering.concepts import ConceptStore
from introspectai.experiments.prefill_authorship import run_authorship_trial
from introspectai.utils.logging import JSONLLogger


def main(
    model_name="mistralai/Ministral-3-8B-Reasoning-2512",
    concepts="sea,tech,food",
    concept_store="datasets/concepts",
    out_dir="datasets/trials",
    layers="10,16,24",
    strengths="1,2,4,8,16",
):
    # Parse args
    if isinstance(concepts, str):
        concepts = concepts.split(",")
    if isinstance(layers, str):
        layers = [int(x) for x in layers.split(",")]
    if isinstance(strengths, str):
        strengths = [float(x) for x in strengths.split(",")]
        
    print(f"Running Experiment B only for model: {model_name}")
    print(f"Concepts: {concepts}")
    print(f"Layers: {layers}")
    print(f"Strengths: {strengths}")
    
    # Load model once
    model, tokenizer = load_model(model_name)
    n_layers = get_layer_count(model)
    print(f"Model has {n_layers} layers")
    
    # Ensure layers are valid
    layers = [l for l in layers if l < n_layers]
    
    # Run Experiment B
    print("\n=== Running Experiment B: Prefill Authorship ===")
    logger_b = JSONLLogger(Path(out_dir) / "sweep_prefill_authorship.jsonl")
    
    for concept in concepts:
        for layer in layers:
            # Load concept vector
            try:
                store = ConceptStore(concept_store)
                vec = store.load(model_name, layer, concept)
                vec = vec.to(model.device).to(model.dtype)
            except (FileNotFoundError, KeyError) as e:
                print(f"Vector not found for {concept} at layer {layer}. Skipping. Error: {e}")
                continue
                
            for strength in strengths:
                print(f"Exp B: Concept {concept}, Layer {layer}, Strength {strength}")
                trial = run_authorship_trial(
                    model=model, 
                    tokenizer=tokenizer, 
                    concept_vector=vec, 
                    layer=layer, 
                    strength=strength, 
                    concept_name=concept
                )
                logger_b.log(trial)
            
    print("\nExperiment B completed!")


if __name__ == "__main__":
    fire.Fire(main)
