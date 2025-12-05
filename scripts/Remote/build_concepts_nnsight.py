import torch
import json
import time
import random
from pathlib import Path
from nnsight import LanguageModel, CONFIG
import numpy as np

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

BASELINE_WORDS_STR = "Desks, Jackets, Gondolas, Laughter, Intelligence, Bicycles, Chairs, Orchestras, Sand, Pottery, Arrowheads, Jewelry, Daffodils, Plateaus, Estuaries, Quilts, Moments, Bamboo, Ravines, Archives, Hieroglyphs, Stars, Clay, Fossils, Wildlife, Flour, Traffic, Bubbles, Honey, Geodes, Magnets, Ribbons, Zigzags, Puzzles, Tornadoes, Anthills, Galaxies, Poverty, Diamonds, Universes, Vinegar, Nebulae, Knowledge, Marble, Fog, Rivers, Scrolls, Silhouettes, Marbles, Cakes, Valleys, Whispers, Pendulums, Towers, Tables, Glaciers, Whirlpools, Jungles, Wool, Anger, Ramparts, Flowers, Research, Hammers, Clouds, Justice, Dogs, Butterflies, Needles, Fortresses, Bonfires, Skyscrapers, Caravans, Patience, Bacon, Velocities, Smoke, Electricity, Sunsets, Anchors, Parchments, Courage, Statues, Oxygen, Time, Butterflies, Fabric, Pasta, Snowflakes, Mountains, Echoes, Pianos, Sanctuaries, Abysses, Air, Dewdrops, Gardens, Literature, Rice, Enigmas"
BASELINE_WORDS = [w.strip().lower() for w in BASELINE_WORDS_STR.split(",")]

CONCEPTS_STR = "Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning, Constellations, Treasures, Phones, Trees, Avalanches, Mirrors, Fountains, Quarries, Sadness, Xylophones, Secrecy, Oceans, Information, Deserts, Kaleidoscopes, Sugar, Vegetables, Poetry, Aquariums, Bags, Peace, Caverns, Memories, Frosts, Volcanoes, Boulders, Harmonies, Masquerades, Rubber, Plastic, Blood, Amphitheaters, Contraptions, Youths, Dynasties, Snow, Dirigibles, Algorithms, Denim, Monoliths, Milk, Bread, Silver, Albert Einstein, Helen Keller, Charles Darwin, Stephen Hawking, Ludwig van Beethoven, Rosa Parks, Thomas Jefferson, Pablo Picasso, William Shakespeare, John F. Kennedy, Benjamin Franklin, Christopher Columbus, Queen Elizabeth II, Marie Curie, Neil Armstrong, Martin Luther King Jr., Genghis Khan, Mother Teresa, Abraham Lincoln, Amelia Earhart, Theodore Roosevelt, Marilyn Monroe, Muhammad Ali, Anne Frank, Joan of Arc, Jane Austen, Aristotle, Michael Jordan, Mahatma Gandhi, Winston Churchill, Frank Sinatra, Nelson Mandela, Vincent van Gogh, Bill Gates, Mark Twain, Charlie Chaplin, Charles Dickens, Franklin D. Roosevelt, Elvis Presley, Isaac Newton, Cleopatra, Joseph Stalin, Julius Caesar, Napoleon Bonaparte, Wolfgang Amadeus Mozart, Galileo Galilei, Alexander the Great, George Washington, Plato, Leonardo da Vinci, Ireland, France, the United Kingdom, New Zealand, Ukraine, Australia, Philippines, North Korea, Pakistan, Russia, Colombia, Thailand, Italy, Spain, South Africa, Morocco, Iran, India, Belgium, Argentina, Brazil, Kenya, Germany, Canada, Japan, Peru, Poland, South Korea, Mexico, Iraq, Ethiopia, Turkey, Bangladesh, the United States, Vietnam, Denmark, Finland, Israel, Switzerland, Indonesia, China, Sweden, Portugal, Egypt, Saudi Arabia, Chile, Greece, Netherlands, Norway, Nigeria, Hats, Radios, Shirts, Trains, Locks, Boxes, Pants, Papers, Windows, Rings, Houses, Chairs, Mirrors, Walls, Necklaces, Books, Batteries, Desks, Bracelets, Keys, Rocks, Computers, Trees, Bottles, Offices, Cameras, Gloves, Coins, Cars, Watches, Buildings, Lamps, Clocks, Bicycles, Speakers, Floors, Phones, Ceilings, Ships, Tables, Apartments, Bridges, Televisions, Shoes, Doors, Needles, Pens, Airplanes, Roads, Pencils, Duty, Evil, Progress, Creativity, Mastery, Competition, Change, Peace, Honor, Good, Unity, Diversity, Trust, Chaos, Liberty, Balance, Harmony, Equality, Conflict, Justice, Ugliness, Morality, Innovation, Power, Space, Tradition, Wisdom, Failure, Democracy, Time, Loyalty, Privilege, Order, Authority, Freedom, Ethics, Cooperation, Independence, Defeat, Truth, Betrayal, Dignity, Success, Courage, Victory, Faith, Knowledge, Rights, Intelligence, Beauty, Thinking, Laughing, Drinking, Singing, Whispering, Reading, Dreaming, Catching, Pulling, Crying, Breathing, Studying, Writing, Screaming, Growing, Talking, Dancing, Falling, Cooking, Winning, Shouting, Learning, Creating, Eating, Pushing, Playing, Teaching, Swimming, Speaking, Destroying, Smiling, Shrinking, Sinking, Breaking, Rising, Floating, Racing, Sleeping, Working, Jumping, Driving, Walking, Flying, Sculpting, Building, Frowning, Striving, Running, Listening, Throwing"
CONCEPTS = [w.strip().lower() for w in CONCEPTS_STR.split(",")]

def get_hash(s):
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()

class NNsightConceptStore:
    def __init__(self, base_path="datasets/concepts_nnsight", model_name="meta-llama/Llama-3.1-405B-Instruct"):
        self.base_path = Path(base_path) / model_name.replace("/", "_").replace("-", "_").lower()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save(self, model_name, layer, concept, vector):
        key = f"{model_name}_{layer}_{concept}"
        h = get_hash(key)
        path = self.base_path / f"{h}.pt"
        torch.save({"vector": vector}, path)
        
        index_path = self.base_path / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    index = json.load(f)
            except:
                index = {}
        else:
            index = {}
        index[key] = str(path)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

def build_concepts_nnsight():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Initializing NNsight model: {model_name}")
    model = LanguageModel(model_name)
    store = NNsightConceptStore(model_name=model_name)
    
    # Dynamic Layer Count
    # NNsight models usually have .config accessible
    if hasattr(model, 'config'):
        n_layers = model.config.num_hidden_layers
    elif hasattr(model.model, 'config'):
        n_layers = model.model.config.num_hidden_layers
    else:
        # Fallback for Llama 3.1 70B if config hidden
        n_layers = 80 
        print(f"Warning: Could not detect layers. Defaulting to {n_layers}")

    layers = list(range(n_layers))
    print(f"Targeting {n_layers} layers.")
    
    # 1. Get Baseline Activations
    print(f"Computing baseline ({len(BASELINE_WORDS)} words)...")
    baseline_acts = {l: [] for l in layers}
    
    for i, word in enumerate(BASELINE_WORDS):
        if i % 10 == 0:
            print(f"Baseline word {i+1}/{len(BASELINE_WORDS)}: {word}")
        
        prompt = f"Tell me about {word}"
        
        # Retry logic for network flakiness
        MAX_RETRIES = 5
        hidden_states = {}
        
        for attempt in range(MAX_RETRIES):
            try:
                print(f"DEBUG: Entering Trace Block. layers len={len(layers)}")
                
                # Use dict to store proxies - must be declared BEFORE with block
                # and populated INSIDE with block
                saved = {}
                with model.trace(prompt, remote=True):
                    for l in layers:
                        saved[l] = model.model.layers[l].output[0][-1, :].save()
                
                # After trace exits, saved dict should have proxies with .value
                hidden_states = saved
                print(f"DEBUG: Trace Done. len(hidden_states)={len(hidden_states)}")
                
                if not hidden_states:
                    raise ValueError("hidden_states is empty after trace!")
                
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to process word '{word}' after {MAX_RETRIES} attempts: {e}")
                    hidden_states = {}
                else:
                    sleep_t = 2 ** attempt + random.uniform(1, 4)
                    print(f"Error processing '{word}': {e}. Retrying in {sleep_t:.1f}s...")
                    time.sleep(sleep_t)
        
        if not hidden_states:
            print(f"DEBUG: hidden_states is empty for word '{word}'")
            continue

        print(f"DEBUG: word '{word}', len(layers)={len(layers)}, len(hidden_states)={len(hidden_states)}")

        # Collect outputs - hidden_states is now a dict keyed by layer
        count = 0
        for l in layers:
            if l in hidden_states:
                baseline_acts[l].append(hidden_states[l])
                count += 1
        print(f"DEBUG: Added {count} activations to baseline_acts")
                
    # 2. Process Baselines
    print("Processing baselines...")
    layer_means = {}
    for layer_idx in layers:
        acts = []
        if len(baseline_acts[layer_idx]) > 0:
             print(f"DEBUG: Layer {layer_idx} has {len(baseline_acts[layer_idx])} captured acts. Checking first item...")
             first_h = baseline_acts[layer_idx][0]
             print(f"DEBUG: Item type: {type(first_h)}, Has .value? {hasattr(first_h, 'value')}")

        for h in baseline_acts[layer_idx]:
            val = h.value if hasattr(h, 'value') else h
            # Handle list vs tensor details if needed, but usually just .value
            if isinstance(val, list):
                 val = torch.tensor(val)
            acts.append(val.unsqueeze(0))
            
        if not acts:
            print(f"WARNING: No acts for layer {layer_idx}")
            continue
            
        all_acts = torch.cat(acts, dim=0)
        layer_means[layer_idx] = all_acts.mean(dim=0).to('cpu') 

    # 3. Get Concept Activations
    print(f"Computing {len(CONCEPTS)} concepts...")
    for i, concept in enumerate(CONCEPTS):
        print(f"Concept {i+1}/{len(CONCEPTS)}: {concept}")
        prompt = f"Tell me about {concept}"
        
        # Unrolled trace block for concept
        hidden_states = {}
        for attempt in range(MAX_RETRIES):
            try:
                saved = {}
                with model.trace(prompt, remote=True):
                    for l in layers:
                        saved[l] = model.model.layers[l].output[0][-1, :].save()
                
                hidden_states = saved
                
                if not hidden_states:
                    raise ValueError("Empty hidden_states after trace")
                    
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed concept '{concept}'. Skipping. Error: {e}")
                    hidden_states = {}
                else:
                    sleep_t = 2 ** attempt + random.uniform(1, 4)
                    print(f"Error concept '{concept}': {e}. Retrying in {sleep_t:.1f}s...")
                    time.sleep(sleep_t)
        
        if not hidden_states:
             continue

        # 4. Compute and Save Vectors - hidden_states is now a dict
        for layer_idx in layers:
            if layer_idx not in hidden_states:
                continue
                
            h = hidden_states[layer_idx]
            if hasattr(h, 'value'):
                c_act = h.value
            else:
                c_act = h
            
            # Ensure tensor
            if not isinstance(c_act, torch.Tensor):
                c_act = torch.tensor(c_act)

            if c_act.dim() > 1:
                c_act = c_act[0]
            
            c_act = c_act.to('cpu')
            
            if layer_idx not in layer_means:
                continue
                
            mean_base = layer_means[layer_idx]
            
            vec = c_act - mean_base
            vec = vec / (torch.norm(vec) + 1e-8)
            
            store.save(model_name, layer_idx, concept, vec)
            
    print("All concepts built and saved.")


if __name__ == "__main__":
    build_concepts_nnsight()
