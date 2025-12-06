
import torch
import json
from pathlib import Path

# Template parts
HEADER = """import torch
import json
from pathlib import Path
from nnsight import LanguageModel, CONFIG
import numpy as np
import time
import random

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

BASELINE_WORDS_STR = "{baseline_str}"
BASELINE_WORDS = [w.strip().lower() for w in BASELINE_WORDS_STR.split(",")]

CONCEPTS_STR = "{concepts_str}"
CONCEPTS = [w.strip().lower() for w in CONCEPTS_STR.split(",")]

BATCH_SIZE = {batch_size}

def get_hash(s):
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()

class NNsightConceptStore:
    def __init__(self, base_path="datasets/concepts_nnsight", model_name="meta-llama/Llama-3.1-405B-Instruct"):
        self.base_path = Path(base_path) / model_name.replace("/", "_").replace("-", "_").lower()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save(self, model_name, layer, concept, vector):
        key = f"{{model_name}}_{{layer}}_{{concept}}"
        h = get_hash(key)
        path = self.base_path / f"{{h}}.pt"
        torch.save({{"vector": vector}}, path)
        
        index_path = self.base_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    index = json.load(f)
            except:
                index = {{}}
        else:
            index = {{}}
        index[key] = str(path)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

def build_concepts_nnsight():
    model_name = "{model_name}"
    print(f"Initializing NNsight model: {{model_name}}")
    model = LanguageModel(model_name)
    store = NNsightConceptStore(model_name=model_name)
    
    # Ensure left padding for correct last-token extraction
    if model.tokenizer.padding_side != "left":
        print("Setting tokenizer padding_side to 'left'")
        model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    
    layers = list(range({n_layers}))
    print(f"Targeting all {n_layers} layers. Batch Size: {{BATCH_SIZE}}")
    
    # --- 1. Get Baseline Activations ---
    print(f"Computing baseline ({{len(BASELINE_WORDS)}} words)...")
    baseline_acts = {{l: [] for l in layers}}
    
    # Chunk baselines
    baseline_chunks = [BASELINE_WORDS[i:i + BATCH_SIZE] for i in range(0, len(BASELINE_WORDS), BATCH_SIZE)]
    
    for batch_idx, batch in enumerate(baseline_chunks):
        print(f"Baseline Batch {{batch_idx+1}}/{{len(baseline_chunks)}} ({{len(batch)}} items)")
        prompts = [f"Tell me about {{word}}" for word in batch]
        
        # Retry Loop
        max_retries = 5
        success = False
        for attempt in range(max_retries):
            try:
                with model.trace(prompts, remote=True):
{unrolled_baseline_trace}
                success = True
                break
            except Exception as e:
                print(f"  Retry {{attempt+1}}/{{max_retries}}: {{e}}")
                time.sleep(2 + random.random() * 5)
        
        if not success:
            print(f"SKIPPING BATCH due to repeated failures.")
            continue

        h0_val = h0.value if hasattr(h0, 'value') else h0 # Check first one
        
        current_batch_size = len(batch)
        layer_proxies = {{}}
{collect_baseline_vars}

        for l in layers:
            h_val = layer_proxies[l].value if hasattr(layer_proxies[l], 'value') else layer_proxies[l]
            
            # h_val is (Batch, Seq, Hidden). need to slice last token.
            if h_val.dim() == 3:
                 h_val = h_val[:, -1, :] # Slice locally!
            elif h_val.dim() == 2:
                 # Logic if nnsight gave us (Batch, Hidden) already? 
                 # Usually output[0] is (B,S,H).
                 pass
            
            # Slice and append
            for b_i in range(current_batch_size):
                if b_i < h_val.size(0):
                    baseline_acts[l].append(h_val[b_i].cpu())

    # --- 2. Compute Means ---
    print("Processing baselines...")
    layer_means = {{}}
    for layer_idx in layers:
        if not baseline_acts[layer_idx]:
             print(f"WARNING: No acts for layer {{layer_idx}}")
             continue
        all_acts = torch.stack(baseline_acts[layer_idx], dim=0) 
        layer_means[layer_idx] = all_acts.mean(dim=0)

    # --- 3. Compute Concepts ---
    print(f"Computing {{len(CONCEPTS)}} concepts...")
    
    concept_chunks = [CONCEPTS[i:i + BATCH_SIZE] for i in range(0, len(CONCEPTS), BATCH_SIZE)]
    
    for batch_idx, batch in enumerate(concept_chunks):
        print(f"Concept Batch {{batch_idx+1}}/{{len(concept_chunks)}} ({{len(batch)}} items)")
        prompts = [f"Tell me about {{word}}" for word in batch]
        
        # Retry Loop
        max_retries = 5
        success = False
        for attempt in range(max_retries):
            try:
                with model.trace(prompts, remote=True):
{unrolled_concept_trace}
                success = True
                break
            except Exception as e:
                print(f"  Retry {{attempt+1}}/{{max_retries}}: {{e}}")
                time.sleep(2 + random.random() * 5)

        if not success:
            print(f"SKIPPING BATCH due to failures.")
            continue

        concept_proxies = {{}}
{collect_concept_vars}
        
        current_batch_size = len(batch)
        for l in layers:
            if l not in layer_means: continue
            
            c_val = concept_proxies[l].value if hasattr(concept_proxies[l], 'value') else concept_proxies[l]
            
            # Slice locally!
            if c_val.dim() == 3:
                 c_val = c_val[:, -1, :]
                 
            mean_base = layer_means[l]
            
            for b_i in range(current_batch_size):
                if b_i < c_val.size(0):
                    vec = c_val[b_i].cpu() - mean_base.cpu()
                    vec = vec / (torch.norm(vec) + 1e-8)
                    store.save(model_name, l, batch[b_i], vec)

    print("All concepts built and saved.")

if __name__ == "__main__":
    build_concepts_nnsight()
"""

# Hardcoded lists
BASELINE_WORDS_STR = "Desks, Jackets, Gondolas, Laughter, Intelligence, Bicycles, Chairs, Orchestras, Sand, Pottery, Arrowheads, Jewelry, Daffodils, Plateaus, Estuaries, Quilts, Moments, Bamboo, Ravines, Archives, Hieroglyphs, Stars, Clay, Fossils, Wildlife, Flour, Traffic, Bubbles, Honey, Geodes, Magnets, Ribbons, Zigzags, Puzzles, Tornadoes, Anthills, Galaxies, Poverty, Diamonds, Universes, Vinegar, Nebulae, Knowledge, Marble, Fog, Rivers, Scrolls, Silhouettes, Marbles, Cakes, Valleys, Whispers, Pendulums, Towers, Tables, Glaciers, Whirlpools, Jungles, Wool, Anger, Ramparts, Flowers, Research, Hammers, Clouds, Justice, Dogs, Butterflies, Needles, Fortresses, Bonfires, Skyscrapers, Caravans, Patience, Bacon, Velocities, Smoke, Electricity, Sunsets, Anchors, Parchments, Courage, Statues, Oxygen, Time, Butterflies, Fabric, Pasta, Snowflakes, Mountains, Echoes, Pianos, Sanctuaries, Abysses, Air, Dewdrops, Gardens, Literature, Rice, Enigmas"
MISC_CONCEPTS_STR = "Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning, Constellations, Treasures, Phones, Trees, Avalanches, Mirrors, Fountains, Quarries, Sadness, Xylophones, Secrecy, Oceans, Information, Deserts, Kaleidoscopes, Sugar, Vegetables, Poetry, Aquariums, Bags, Peace, Caverns, Memories, Frosts, Volcanoes, Boulders, Harmonies, Masquerades, Rubber, Plastic, Blood, Amphitheaters, Contraptions, Youths, Dynasties, Snow, Dirigibles, Algorithms, Denim, Monoliths, Milk, Bread, Silver"
PEOPLE_CONCEPTS_STR = "Albert Einstein, Helen Keller, Charles Darwin, Stephen Hawking, Ludwig van Beethoven, Rosa Parks, Thomas Jefferson, Pablo Picasso, William Shakespeare, John F. Kennedy, Benjamin Franklin, Christopher Columbus, Queen Elizabeth II, Marie Curie, Neil Armstrong, Martin Luther King Jr., Genghis Khan, Mother Teresa, Abraham Lincoln, Amelia Earhart, Theodore Roosevelt, Marilyn Monroe, Muhammad Ali, Anne Frank, Joan of Arc, Jane Austen, Aristotle, Michael Jordan, Mahatma Gandhi, Winston Churchill, Frank Sinatra, Nelson Mandela, Vincent van Gogh, Bill Gates, Mark Twain, Charlie Chaplin, Charles Dickens, Franklin D. Roosevelt, Elvis Presley, Isaac Newton, Cleopatra, Joseph Stalin, Julius Caesar, Napoleon Bonaparte, Wolfgang Amadeus Mozart, Galileo Galilei, Alexander the Great, George Washington, Plato, Leonardo da Vinci"
COUNTRIES_CONCEPTS_STR = "Ireland, France, the United Kingdom, New Zealand, Ukraine, Australia, Philippines, North Korea, Pakistan, Russia, Colombia, Thailand, Italy, Spain, South Africa, Morocco, Iran, India, Belgium, Argentina, Brazil, Kenya, Germany, Canada, Japan, Peru, Poland, South Korea, Mexico, Iraq, Ethiopia, Turkey, Bangladesh, the United States, Vietnam, Denmark, Finland, Israel, Switzerland, Indonesia, China, Sweden, Portugal, Egypt, Saudi Arabia, Chile, Greece, Netherlands, Norway, Nigeria"
OBJECTS_CONCEPTS_STR = "Hats, Radios, Shirts, Trains, Locks, Boxes, Pants, Papers, Windows, Rings, Houses, Chairs, Mirrors, Walls, Necklaces, Books, Batteries, Desks, Bracelets, Keys, Rocks, Computers, Trees, Bottles, Offices, Cameras, Gloves, Coins, Cars, Watches, Buildings, Lamps, Clocks, Bicycles, Speakers, Floors, Phones, Ceilings, Ships, Tables, Apartments, Bridges, Televisions, Shoes, Doors, Needles, Pens, Airplanes, Roads, Pencils"
ABSTRACT_CONCEPTS_STR = "Duty, Evil, Progress, Creativity, Mastery, Competition, Change, Peace, Honor, Good, Unity, Diversity, Trust, Chaos, Liberty, Balance, Harmony, Equality, Conflict, Justice, Ugliness, Morality, Innovation, Power, Space, Tradition, Wisdom, Failure, Democracy, Time, Loyalty, Privilege, Order, Authority, Freedom, Ethics, Cooperation, Independence, Defeat, Truth, Betrayal, Dignity, Success, Courage, Victory, Faith, Knowledge, Rights, Intelligence, Beauty"
ACTIONS_CONCEPTS_STR = "Thinking, Laughing, Drinking, Singing, Whispering, Reading, Dreaming, Catching, Pulling, Crying, Breathing, Studying, Writing, Screaming, Growing, Talking, Dancing, Falling, Cooking, Winning, Shouting, Learning, Creating, Eating, Pushing, Playing, Teaching, Swimming, Speaking, Destroying, Smiling, Shrinking, Sinking, Breaking, Rising, Floating, Racing, Sleeping, Working, Jumping, Driving, Walking, Flying, Sculpting, Building, Frowning, Striving, Running, Listening, Throwing"

CONCEPTS_STR = f"{MISC_CONCEPTS_STR}, {PEOPLE_CONCEPTS_STR}, {COUNTRIES_CONCEPTS_STR}, {OBJECTS_CONCEPTS_STR}, {ABSTRACT_CONCEPTS_STR}, {ACTIONS_CONCEPTS_STR}"

def generate_script(model_name="meta-llama/Llama-3.1-70B-Instruct", n_layers=80):
    BATCH_SIZE = 32
    
    # 1. Unrolled Baseline Trace (h0 = ... output[0].save()) <-- NO SLICING HERE
    unrolled_baseline_trace = ""
    for l in range(n_layers):
        unrolled_baseline_trace += f"                    h{l} = model.model.layers[{l}].output[0].save()\n"
        
    # 2. Collect Baseline Vars (layer_proxies[0] = h0)
    collect_baseline_vars = ""
    for l in range(n_layers):
        collect_baseline_vars += f"        layer_proxies[{l}] = h{l}\n"
        
    # 3. Unrolled Concept Trace (c0 = ... output[0].save()) <-- NO SLICING HERE
    unrolled_concept_trace = ""
    for l in range(n_layers):
        unrolled_concept_trace += f"                    c{l} = model.model.layers[{l}].output[0].save()\n"
        
    # 4. Collect Concept Vars
    collect_concept_vars = ""
    for l in range(n_layers):
        collect_concept_vars += f"        concept_proxies[{l}] = c{l}\n"
        
    # Format Code
    code = HEADER.format(
        baseline_str=BASELINE_WORDS_STR,
        concepts_str=CONCEPTS_STR,
        model_name=model_name,
        n_layers=n_layers,
        batch_size=BATCH_SIZE,
        unrolled_baseline_trace=unrolled_baseline_trace,
        collect_baseline_vars=collect_baseline_vars,
        unrolled_concept_trace=unrolled_concept_trace,
        collect_concept_vars=collect_concept_vars
    )
    
    model_suffix = model_name.split("/")[-1].replace("-", "_").lower()
    output_path = f"scripts/Remote/build_concepts_nnsight_{model_suffix}.py"
    
    with open(output_path, "w") as f:
        f.write(code)
    print(f"Generated {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--layers", type=int, default=80)
    args = parser.parse_args()
    
    generate_script(args.model, args.layers)
