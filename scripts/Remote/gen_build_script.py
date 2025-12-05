
BASELINE_WORDS_STR = "Desks, Jackets, Gondolas, Laughter, Intelligence, Bicycles, Chairs, Orchestras, Sand, Pottery, Arrowheads, Jewelry, Daffodils, Plateaus, Estuaries, Quilts, Moments, Bamboo, Ravines, Archives, Hieroglyphs, Stars, Clay, Fossils, Wildlife, Flour, Traffic, Bubbles, Honey, Geodes, Magnets, Ribbons, Zigzags, Puzzles, Tornadoes, Anthills, Galaxies, Poverty, Diamonds, Universes, Vinegar, Nebulae, Knowledge, Marble, Fog, Rivers, Scrolls, Silhouettes, Marbles, Cakes, Valleys, Whispers, Pendulums, Towers, Tables, Glaciers, Whirlpools, Jungles, Wool, Anger, Ramparts, Flowers, Research, Hammers, Clouds, Justice, Dogs, Butterflies, Needles, Fortresses, Bonfires, Skyscrapers, Caravans, Patience, Bacon, Velocities, Smoke, Electricity, Sunsets, Anchors, Parchments, Courage, Statues, Oxygen, Time, Butterflies, Fabric, Pasta, Snowflakes, Mountains, Echoes, Pianos, Sanctuaries, Abysses, Air, Dewdrops, Gardens, Literature, Rice, Enigmas"

MISC_CONCEPTS_STR = "Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning, Constellations, Treasures, Phones, Trees, Avalanches, Mirrors, Fountains, Quarries, Sadness, Xylophones, Secrecy, Oceans, Information, Deserts, Kaleidoscopes, Sugar, Vegetables, Poetry, Aquariums, Bags, Peace, Caverns, Memories, Frosts, Volcanoes, Boulders, Harmonies, Masquerades, Rubber, Plastic, Blood, Amphitheaters, Contraptions, Youths, Dynasties, Snow, Dirigibles, Algorithms, Denim, Monoliths, Milk, Bread, Silver"

PEOPLE_CONCEPTS_STR = "Albert Einstein, Helen Keller, Charles Darwin, Stephen Hawking, Ludwig van Beethoven, Rosa Parks, Thomas Jefferson, Pablo Picasso, William Shakespeare, John F. Kennedy, Benjamin Franklin, Christopher Columbus, Queen Elizabeth II, Marie Curie, Neil Armstrong, Martin Luther King Jr., Genghis Khan, Mother Teresa, Abraham Lincoln, Amelia Earhart, Theodore Roosevelt, Marilyn Monroe, Muhammad Ali, Anne Frank, Joan of Arc, Jane Austen, Aristotle, Michael Jordan, Mahatma Gandhi, Winston Churchill, Frank Sinatra, Nelson Mandela, Vincent van Gogh, Bill Gates, Mark Twain, Charlie Chaplin, Charles Dickens, Franklin D. Roosevelt, Elvis Presley, Isaac Newton, Cleopatra, Joseph Stalin, Julius Caesar, Napoleon Bonaparte, Wolfgang Amadeus Mozart, Galileo Galilei, Alexander the Great, George Washington, Plato, Leonardo da Vinci"

COUNTRIES_CONCEPTS_STR = "Ireland, France, the United Kingdom, New Zealand, Ukraine, Australia, Philippines, North Korea, Pakistan, Russia, Colombia, Thailand, Italy, Spain, South Africa, Morocco, Iran, India, Belgium, Argentina, Brazil, Kenya, Germany, Canada, Japan, Peru, Poland, South Korea, Mexico, Iraq, Ethiopia, Turkey, Bangladesh, the United States, Vietnam, Denmark, Finland, Israel, Switzerland, Indonesia, China, Sweden, Portugal, Egypt, Saudi Arabia, Chile, Greece, Netherlands, Norway, Nigeria"

OBJECTS_CONCEPTS_STR = "Hats, Radios, Shirts, Trains, Locks, Boxes, Pants, Papers, Windows, Rings, Houses, Chairs, Mirrors, Walls, Necklaces, Books, Batteries, Desks, Bracelets, Keys, Rocks, Computers, Trees, Bottles, Offices, Cameras, Gloves, Coins, Cars, Watches, Buildings, Lamps, Clocks, Bicycles, Speakers, Floors, Phones, Ceilings, Ships, Tables, Apartments, Bridges, Televisions, Shoes, Doors, Needles, Pens, Airplanes, Roads, Pencils"

ABSTRACT_CONCEPTS_STR = "Duty, Evil, Progress, Creativity, Mastery, Competition, Change, Peace, Honor, Good, Unity, Diversity, Trust, Chaos, Liberty, Balance, Harmony, Equality, Conflict, Justice, Ugliness, Morality, Innovation, Power, Space, Tradition, Wisdom, Failure, Democracy, Time, Loyalty, Privilege, Order, Authority, Freedom, Ethics, Cooperation, Independence, Defeat, Truth, Betrayal, Dignity, Success, Courage, Victory, Faith, Knowledge, Rights, Intelligence, Beauty"

ACTIONS_CONCEPTS_STR = "Thinking, Laughing, Drinking, Singing, Whispering, Reading, Dreaming, Catching, Pulling, Crying, Breathing, Studying, Writing, Screaming, Growing, Talking, Dancing, Falling, Cooking, Winning, Shouting, Learning, Creating, Eating, Pushing, Playing, Teaching, Swimming, Speaking, Destroying, Smiling, Shrinking, Sinking, Breaking, Rising, Floating, Racing, Sleeping, Working, Jumping, Driving, Walking, Flying, Sculpting, Building, Frowning, Striving, Running, Listening, Throwing"

CONCEPTS_STR = f"{MISC_CONCEPTS_STR}, {PEOPLE_CONCEPTS_STR}, {COUNTRIES_CONCEPTS_STR}, {OBJECTS_CONCEPTS_STR}, {ABSTRACT_CONCEPTS_STR}, {ACTIONS_CONCEPTS_STR}"

HEADER = """import torch
import json
from pathlib import Path
from nnsight import LanguageModel, CONFIG
import numpy as np

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

BASELINE_WORDS_STR = "{baseline_str}"
BASELINE_WORDS = [w.strip().lower() for w in BASELINE_WORDS_STR.split(",")]

CONCEPTS_STR = "{concepts_str}"
CONCEPTS = [w.strip().lower() for w in CONCEPTS_STR.split(",")]

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
    model_name = "meta-llama/Llama-3.1-405B-Instruct"
    print(f"Initializing NNsight model: {{model_name}}")
    model = LanguageModel(model_name)
    store = NNsightConceptStore(model_name=model_name)
    
    layers = list(range(126))
    print(f"Targeting all 126 layers.")
    
    # 1. Get Baseline Activations
    print(f"Computing baseline ({{len(BASELINE_WORDS)}} words)...")
    baseline_acts = {{l: [] for l in layers}}
    
    for i, word in enumerate(BASELINE_WORDS):
        if i % 10 == 0:
            print(f"Baseline word {{i+1}}/{{len(BASELINE_WORDS)}}: {{word}}")
            
        prompt = f"Tell me about {{word}}"
        
        # Unrolled trace block
        with model.trace(prompt, remote=True):
"""

FOOTER = """
        # Collect outputs
        layer_outputs = {}
"""

PROCESS_BASELINES = """
        # Check if captured
        if not layer_outputs:
            print(f"WARNING: No layer outputs captured for {{word}}!")
            continue

        for l in layers:
            baseline_acts[l].append(layer_outputs[l])
                
    # 2. Process Baselines
    print("Processing baselines...")
    layer_means = {}
    for layer_idx in layers:
        acts = []
        for h in baseline_acts[layer_idx]:
            val = h.value if hasattr(h, 'value') else h
            acts.append(val.unsqueeze(0))
        if not acts:
            print(f"WARNING: No acts for layer {{layer_idx}}")
            continue
        all_acts = torch.cat(acts, dim=0)
        layer_means[layer_idx] = all_acts.mean(dim=0)

    # 3. Get Concept Activations
    print(f"Computing {{len(CONCEPTS)}} concepts...")
    for i, concept in enumerate(CONCEPTS):
        print(f"Concept {{i+1}}/{{len(CONCEPTS)}}: {{concept}}")
        prompt = f"Tell me about {{concept}}"
        
        # Unrolled trace block for concept
        with model.trace(prompt, remote=True):
"""

SAVE_CONCEPTS = """
        # Collect outputs
        concept_acts_proxies = {}
"""

FINISH = """
        # 4. Compute and Save Vectors
        for layer_idx in layers:
            h = concept_acts_proxies[layer_idx]
            if hasattr(h, 'value'):
                c_act = h.value
            else:
                c_act = h
            
            if c_act.dim() > 1:
                c_act = c_act[0]
            
            if layer_idx not in layer_means:
                continue
                
            mean_base = layer_means[layer_idx]
            
            vec = c_act - mean_base
            vec = vec / (torch.norm(vec) + 1e-8)
            
            store.save(model_name, layer_idx, concept, vec)
            
    print("All concepts built and saved.")

if __name__ == "__main__":
    build_concepts_nnsight()
"""

def generate_script(model_name="meta-llama/Llama-3.1-70B-Instruct", n_layers=80):
    # Update header for dynamic model
    header = HEADER.format(baseline_str=BASELINE_WORDS_STR, concepts_str=CONCEPTS_STR)
    header = header.replace('model_name = "meta-llama/Llama-3.1-405B-Instruct"', f'model_name = "{model_name}"')
    header = header.replace('layers = list(range(126))', f'layers = list(range({n_layers}))')
    header = header.replace('Targeting all 126 layers', f'Targeting all {n_layers} layers')
    
    script = header
    
    # Unroll baseline trace
    for l in range(n_layers):
        script += f"            h{l} = model.model.layers[{l}].output[0][-1, :].save()\n"
    
    script += FOOTER
    
    # Collect baseline outputs
    for l in range(n_layers):
        script += f"        layer_outputs[{l}] = h{l}\n"
        
    script += PROCESS_BASELINES.replace('range(126)', f'range({n_layers})')
    
    # Unroll concept trace
    for l in range(n_layers):
        script += f"            c{l} = model.model.layers[{l}].output[0][-1, :].save()\n"
        
    script += SAVE_CONCEPTS
    
    # Collect concept outputs
    for l in range(n_layers):
        script += f"        concept_acts_proxies[{l}] = c{l}\n"
        
    script += FINISH
    
    model_suffix = model_name.split("/")[-1].replace("-", "_").lower()
    output_path = f"scripts/Remote/build_concepts_nnsight_{model_suffix}.py"
    
    with open(output_path, "w") as f:
        f.write(script)
    print(f"Generated {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--layers", type=int, default=80)
    args = parser.parse_args()
    
    generate_script(args.model, args.layers)
