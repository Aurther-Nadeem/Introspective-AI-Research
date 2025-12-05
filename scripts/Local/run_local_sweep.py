import torch
import fire
import json
import uuid
import random
from pathlib import Path
from introspectai.models.load import load_model, get_layer_count
from introspectai.steering.concepts import build_concept_vector_from_word, ConceptStore
from introspectai.experiments.injected_thoughts import run_trial as run_trial_a
from introspectai.experiments.prefill_authorship import run_authorship_trial as run_trial_b
from introspectai.utils.logging import JSONLLogger
from introspectai.eval.grader import HeuristicGrader
from introspectai.eval.llm_grader import LLMGrader

from introspectai.experiments.distinguish_thoughts import run_distinguish_trial
from introspectai.experiments.intended_vs_unintended import run_intended_trial
from introspectai.experiments.intentional_control import run_control_trial, PROMPT_TEMPLATES

def main(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    concepts=None,
    concept_store="datasets/concepts",
    out_dir=None,
    layers="10,16,24",
    strengths=None,
    context_lengths="0",
    grader="heuristic", # 'heuristic' or 'llm'
    api_key=None,
    n_trials=50,
    experiments="A,B" # A, B, C, D, E or "all"
):
    # Defaults and Type Handling
    if layers is None:
        layers = [10, 16, 24]
    elif isinstance(layers, str):
        layers = [int(x) for x in layers.split(",")]
    elif isinstance(layers, int):
        layers = [layers]
        
    if strengths is None:
        strengths = [5.0, 10.0, 15.0, 20.0]
    elif isinstance(strengths, str):
        strengths = [float(x) for x in strengths.split(",")]
    elif isinstance(strengths, (int, float)):
        strengths = [float(strengths)]
        
    if concepts is None:
        concepts = ["dust", "satellites", "trumpets", "origami"]
    elif isinstance(concepts, str):
        concepts = concepts.split(",")
        
    if isinstance(context_lengths, str):
        context_lengths = [int(x) for x in context_lengths.split(",")]
    elif isinstance(context_lengths, int):
        context_lengths = [context_lengths]

    if isinstance(experiments, str):
        if experiments.lower() == "all":
            experiments = ["A", "B", "C", "D", "E"]
        else:
            experiments = [x.strip().upper() for x in experiments.split(",")]
    elif isinstance(experiments, (list, tuple)):
        experiments = [str(x).strip().upper() for x in experiments]
        
    print(f"Running sweep for model: {model_name}")
    print(f"Experiments: {experiments}")
    print(f"Concepts: {concepts}")
    print(f"Layers: {layers}")
    print(f"Strengths: {strengths}")
    print(f"Context Lengths: {context_lengths}")
    
    # Handle output directory
    if out_dir is None:
        slug = model_name.split("/")[-1].lower().replace("-", "_")
        out_dir = f"datasets/trials/{slug}"
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_path}")

    # Initialize Grader (only needed for Exp A)
    grader_obj = None
    if "A" in experiments:
        if grader == "llm":
            print("Using LLM Grader (API)")
            grader_obj = LLMGrader(api_key=api_key)
        else:
            print("Using Heuristic Grader (Regex)")
            grader_obj = HeuristicGrader()

    # Load model once
    model, tokenizer = load_model(model_name)
    n_layers = get_layer_count(model)
    print(f"Model has {n_layers} layers")
    
    # Ensure layers are valid
    layers = [l for l in layers if l < n_layers]
    
    # 1. Build Concepts (Always check/build if needed)
    print("\n=== Building Concepts (Paper Method) ===")
    
    # Ensure random words exist
    random_words_file = "datasets/random_words.txt"
    if not Path(random_words_file).exists():
        words = ["apple", "chair", "history", "math", "river", "cloud", "idea", "person", "time", "way", 
                 "year", "work", "government", "day", "man", "world", "life", "part", "house", "course",
                 "case", "system", "place", "end", "group", "company", "party", "information", "school", "fact",
                 "money", "point", "example", "state", "business", "night", "area", "water", "thing", "family",
                 "head", "hand", "order", "john", "side", "home", "development", "week", "power", "country"]
        Path(random_words_file).write_text("\n".join(words))
    
    random_words = Path(random_words_file).read_text().strip().splitlines()
    
    from introspectai.steering.concepts import build_concept_vector_from_word, ConceptStore
    
    # Check if vectors exist, if not build them
    store = ConceptStore(concept_store)
    for concept in concepts:
        for layer in layers:
            try:
                store.load(model_name, layer, concept)
            except (FileNotFoundError, KeyError):
                print(f"Building concept '{concept}' at layer {layer}...")
                vec, stats = build_concept_vector_from_word(
                    model, tokenizer, layer, concept, random_words
                )
                store.save(model_name, layer, concept, vec, stats)
        
    # 2. Run Experiment A: Injected Thoughts
    if "A" in experiments:
        print("\n=== Running Experiment A: Injected Thoughts ===")
        logger = JSONLLogger(Path(out_dir) / "sweep_injected_thoughts.jsonl")
        
        for concept in concepts:
            store = ConceptStore(concept_store)
            for layer in layers:
                try:
                    vec = store.load(model_name, layer, concept)
                except FileNotFoundError:
                    continue
                    
                for strength in strengths:
                    print(f"Exp A: Concept {concept}, Layer {layer}, Strength {strength}")
                    for _ in range(n_trials):
                        result = run_trial_a(
                            model, tokenizer, 
                            concept_vec=vec, 
                            layer_idx=layer, 
                            strength=strength, 
                            seed=random.randint(0, 100000)
                        )
                        graded = grader_obj.grade(
                            response=result["completion"], 
                            target_concept=concept if result["injected"] else None,
                            prompt_text=result["text"].replace(result["completion"], "")
                        )
                        log_entry = {
                            "trial_id": uuid.uuid4().hex,
                            "experiment": "injected_thoughts",
                            "injected": result["injected"],
                            "concept": concept if result["injected"] else None,
                            "layer": layer if result["injected"] else None,
                            "strength": strength if result["injected"] else None,
                            "prompt_type": result["prompt_type"],
                            "text": result["text"],
                            "completion": result["completion"],
                            "parsed": graded
                        }
                        logger.log(log_entry)

    # 3. Run Experiment B: Prefill Authorship
    if "B" in experiments:
        print("\n=== Running Experiment B: Prefill Authorship ===")
        logger_b = JSONLLogger(Path(out_dir) / "sweep_prefill_authorship.jsonl")
        
        for concept in concepts:
            for layer in layers:
                try:
                    store = ConceptStore(concept_store)
                    vec = store.load(model_name, layer, concept)
                    vec = vec.to(model.device).to(model.dtype)
                except (FileNotFoundError, KeyError):
                    continue
                    
                for strength in strengths:
                    print(f"Exp B: Concept {concept}, Layer {layer}, Strength {strength}")
                    trial = run_trial_b(
                        model=model, 
                        tokenizer=tokenizer, 
                        concept_vector=vec, 
                        layer=layer, 
                        strength=strength, 
                        concept_name=concept
                    )
                    logger_b.log(trial)

    # 4. Run Experiment C: Distinguishing Thoughts
    if "C" in experiments:
        print("\n=== Running Experiment C: Distinguishing Thoughts ===")
        logger_c = JSONLLogger(Path(out_dir) / "sweep_distinguish.jsonl")
        
        for concept in concepts:
            for layer in layers:
                try:
                    store = ConceptStore(concept_store)
                    vec = store.load(model_name, layer, concept)
                    vec = vec.to(model.device).to(model.dtype)
                except (FileNotFoundError, KeyError):
                    continue
                    
                for strength in strengths:
                    print(f"Exp C: Concept {concept}, Layer {layer}, Strength {strength}")
                    for _ in range(n_trials):
                        for p_type in ["repeat", "think"]:
                            try:
                                result = run_distinguish_trial(
                                    model, tokenizer, vec, layer, strength, concept, prompt_type=p_type
                                )
                                result["model"] = model_name
                                logger_c.log(result)
                            except Exception as e:
                                print(f"Error in Exp C trial: {e}")

    # 5. Run Experiment D: Intended vs Unintended
    if "D" in experiments:
        print("\n=== Running Experiment D: Intended vs Unintended ===")
        logger_d = JSONLLogger(Path(out_dir) / "sweep_intended.jsonl")
        
        for concept in concepts:
            for layer in layers:
                try:
                    store = ConceptStore(concept_store)
                    vec = store.load(model_name, layer, concept)
                    vec = vec.to(model.device).to(model.dtype)
                except (FileNotFoundError, KeyError):
                    continue
                    
                for strength in strengths:
                    print(f"Exp D: Concept {concept}, Layer {layer}, Strength {strength}")
                    for _ in range(n_trials):
                        try:
                            result = run_intended_trial(
                                model, tokenizer, vec, layer, strength, concept
                            )
                            result["model"] = model_name
                            logger_d.log(result)
                        except Exception as e:
                            print(f"Error in Exp D trial: {e}")

    # 6. Run Experiment E: Intentional Control
    if "E" in experiments:
        print("\n=== Running Experiment E: Intentional Control ===")
        logger_e = JSONLLogger(Path(out_dir) / "sweep_control.jsonl")
        conditions = list(PROMPT_TEMPLATES.keys())
        
        for concept in concepts:
            for layer in layers:
                try:
                    store = ConceptStore(concept_store)
                    vec = store.load(model_name, layer, concept)
                    vec = vec.to(model.device).to(model.dtype)
                except (FileNotFoundError, KeyError):
                    continue
                    
                for condition in conditions:
                    # Note: Control sweep usually doesn't sweep strength, it measures control.
                    # But if we want to see if injection *helps* control, we might inject?
                    # Wait, Exp E in the paper is about *measuring* control (cosine sim) without injection,
                    # OR injecting to see if it *improves* control.
                    # The `run_control_trial` function takes a vector but uses it for *measurement* (cosine sim target).
                    # It does NOT inject by default unless we modify it.
                    # Let's check `run_control_trial` signature.
                    # run_control_trial(model, tokenizer, concept_vector, layer_idx, concept_name, condition="positive")
                    # It uses the vector to calculate cosine similarity of the output.
                    # It does NOT take strength. So we loop conditions, not strengths.
                    
                    print(f"Exp E: Concept {concept}, Layer {layer}, Condition {condition}")
                    for _ in range(n_trials):
                        try:
                            result = run_control_trial(
                                model, tokenizer, vec, layer, concept, condition=condition
                            )
                            result["model"] = model_name
                            logger_e.log(result)
                        except Exception as e:
                            print(f"Error in Exp E trial: {e}")

    print("\nSweep completed!")

if __name__ == "__main__":
    fire.Fire(main)
