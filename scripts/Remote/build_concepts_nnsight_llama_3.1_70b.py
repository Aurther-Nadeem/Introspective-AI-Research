import torch
import json
from pathlib import Path
from nnsight import LanguageModel, CONFIG
import numpy as np
import time
import random

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

BASELINE_WORDS_STR = "Desks, Jackets, Gondolas, Laughter, Intelligence, Bicycles, Chairs, Orchestras, Sand, Pottery, Arrowheads, Jewelry, Daffodils, Plateaus, Estuaries, Quilts, Moments, Bamboo, Ravines, Archives, Hieroglyphs, Stars, Clay, Fossils, Wildlife, Flour, Traffic, Bubbles, Honey, Geodes, Magnets, Ribbons, Zigzags, Puzzles, Tornadoes, Anthills, Galaxies, Poverty, Diamonds, Universes, Vinegar, Nebulae, Knowledge, Marble, Fog, Rivers, Scrolls, Silhouettes, Marbles, Cakes, Valleys, Whispers, Pendulums, Towers, Tables, Glaciers, Whirlpools, Jungles, Wool, Anger, Ramparts, Flowers, Research, Hammers, Clouds, Justice, Dogs, Butterflies, Needles, Fortresses, Bonfires, Skyscrapers, Caravans, Patience, Bacon, Velocities, Smoke, Electricity, Sunsets, Anchors, Parchments, Courage, Statues, Oxygen, Time, Butterflies, Fabric, Pasta, Snowflakes, Mountains, Echoes, Pianos, Sanctuaries, Abysses, Air, Dewdrops, Gardens, Literature, Rice, Enigmas"
BASELINE_WORDS = [w.strip().lower() for w in BASELINE_WORDS_STR.split(",")]

CONCEPTS_STR = "Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning, Constellations, Treasures, Phones, Trees, Avalanches, Mirrors, Fountains, Quarries, Sadness, Xylophones, Secrecy, Oceans, Information, Deserts, Kaleidoscopes, Sugar, Vegetables, Poetry, Aquariums, Bags, Peace, Caverns, Memories, Frosts, Volcanoes, Boulders, Harmonies, Masquerades, Rubber, Plastic, Blood, Amphitheaters, Contraptions, Youths, Dynasties, Snow, Dirigibles, Algorithms, Denim, Monoliths, Milk, Bread, Silver, Albert Einstein, Helen Keller, Charles Darwin, Stephen Hawking, Ludwig van Beethoven, Rosa Parks, Thomas Jefferson, Pablo Picasso, William Shakespeare, John F. Kennedy, Benjamin Franklin, Christopher Columbus, Queen Elizabeth II, Marie Curie, Neil Armstrong, Martin Luther King Jr., Genghis Khan, Mother Teresa, Abraham Lincoln, Amelia Earhart, Theodore Roosevelt, Marilyn Monroe, Muhammad Ali, Anne Frank, Joan of Arc, Jane Austen, Aristotle, Michael Jordan, Mahatma Gandhi, Winston Churchill, Frank Sinatra, Nelson Mandela, Vincent van Gogh, Bill Gates, Mark Twain, Charlie Chaplin, Charles Dickens, Franklin D. Roosevelt, Elvis Presley, Isaac Newton, Cleopatra, Joseph Stalin, Julius Caesar, Napoleon Bonaparte, Wolfgang Amadeus Mozart, Galileo Galilei, Alexander the Great, George Washington, Plato, Leonardo da Vinci, Ireland, France, the United Kingdom, New Zealand, Ukraine, Australia, Philippines, North Korea, Pakistan, Russia, Colombia, Thailand, Italy, Spain, South Africa, Morocco, Iran, India, Belgium, Argentina, Brazil, Kenya, Germany, Canada, Japan, Peru, Poland, South Korea, Mexico, Iraq, Ethiopia, Turkey, Bangladesh, the United States, Vietnam, Denmark, Finland, Israel, Switzerland, Indonesia, China, Sweden, Portugal, Egypt, Saudi Arabia, Chile, Greece, Netherlands, Norway, Nigeria, Hats, Radios, Shirts, Trains, Locks, Boxes, Pants, Papers, Windows, Rings, Houses, Chairs, Mirrors, Walls, Necklaces, Books, Batteries, Desks, Bracelets, Keys, Rocks, Computers, Trees, Bottles, Offices, Cameras, Gloves, Coins, Cars, Watches, Buildings, Lamps, Clocks, Bicycles, Speakers, Floors, Phones, Ceilings, Ships, Tables, Apartments, Bridges, Televisions, Shoes, Doors, Needles, Pens, Airplanes, Roads, Pencils, Duty, Evil, Progress, Creativity, Mastery, Competition, Change, Peace, Honor, Good, Unity, Diversity, Trust, Chaos, Liberty, Balance, Harmony, Equality, Conflict, Justice, Ugliness, Morality, Innovation, Power, Space, Tradition, Wisdom, Failure, Democracy, Time, Loyalty, Privilege, Order, Authority, Freedom, Ethics, Cooperation, Independence, Defeat, Truth, Betrayal, Dignity, Success, Courage, Victory, Faith, Knowledge, Rights, Intelligence, Beauty, Thinking, Laughing, Drinking, Singing, Whispering, Reading, Dreaming, Catching, Pulling, Crying, Breathing, Studying, Writing, Screaming, Growing, Talking, Dancing, Falling, Cooking, Winning, Shouting, Learning, Creating, Eating, Pushing, Playing, Teaching, Swimming, Speaking, Destroying, Smiling, Shrinking, Sinking, Breaking, Rising, Floating, Racing, Sleeping, Working, Jumping, Driving, Walking, Flying, Sculpting, Building, Frowning, Striving, Running, Listening, Throwing"
CONCEPTS = [w.strip().lower() for w in CONCEPTS_STR.split(",")]

BATCH_SIZE = 32

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
    model_name = "meta-llama/Llama-3.1-70B"
    print(f"Initializing NNsight model: {model_name}")
    model = LanguageModel(model_name)
    store = NNsightConceptStore(model_name=model_name)
    
    # Ensure left padding for correct last-token extraction
    if model.tokenizer.padding_side != "left":
        print("Setting tokenizer padding_side to 'left'")
        model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    
    layers = list(range(80))
    print(f"Targeting all 80 layers. Batch Size: {BATCH_SIZE}")
    
    # --- 1. Get Baseline Activations ---
    print(f"Computing baseline ({len(BASELINE_WORDS)} words)...")
    baseline_acts = {l: [] for l in layers}
    
    # Chunk baselines
    baseline_chunks = [BASELINE_WORDS[i:i + BATCH_SIZE] for i in range(0, len(BASELINE_WORDS), BATCH_SIZE)]
    
    for batch_idx, batch in enumerate(baseline_chunks):
        print(f"Baseline Batch {batch_idx+1}/{len(baseline_chunks)} ({len(batch)} items)")
        prompts = [f"Tell me about {word}" for word in batch]
        
        # Retry Loop
        max_retries = 5
        success = False
        for attempt in range(max_retries):
            try:
                with model.trace(prompts, remote=True):
                    h0 = model.model.layers[0].output[0].save()
                    h1 = model.model.layers[1].output[0].save()
                    h2 = model.model.layers[2].output[0].save()
                    h3 = model.model.layers[3].output[0].save()
                    h4 = model.model.layers[4].output[0].save()
                    h5 = model.model.layers[5].output[0].save()
                    h6 = model.model.layers[6].output[0].save()
                    h7 = model.model.layers[7].output[0].save()
                    h8 = model.model.layers[8].output[0].save()
                    h9 = model.model.layers[9].output[0].save()
                    h10 = model.model.layers[10].output[0].save()
                    h11 = model.model.layers[11].output[0].save()
                    h12 = model.model.layers[12].output[0].save()
                    h13 = model.model.layers[13].output[0].save()
                    h14 = model.model.layers[14].output[0].save()
                    h15 = model.model.layers[15].output[0].save()
                    h16 = model.model.layers[16].output[0].save()
                    h17 = model.model.layers[17].output[0].save()
                    h18 = model.model.layers[18].output[0].save()
                    h19 = model.model.layers[19].output[0].save()
                    h20 = model.model.layers[20].output[0].save()
                    h21 = model.model.layers[21].output[0].save()
                    h22 = model.model.layers[22].output[0].save()
                    h23 = model.model.layers[23].output[0].save()
                    h24 = model.model.layers[24].output[0].save()
                    h25 = model.model.layers[25].output[0].save()
                    h26 = model.model.layers[26].output[0].save()
                    h27 = model.model.layers[27].output[0].save()
                    h28 = model.model.layers[28].output[0].save()
                    h29 = model.model.layers[29].output[0].save()
                    h30 = model.model.layers[30].output[0].save()
                    h31 = model.model.layers[31].output[0].save()
                    h32 = model.model.layers[32].output[0].save()
                    h33 = model.model.layers[33].output[0].save()
                    h34 = model.model.layers[34].output[0].save()
                    h35 = model.model.layers[35].output[0].save()
                    h36 = model.model.layers[36].output[0].save()
                    h37 = model.model.layers[37].output[0].save()
                    h38 = model.model.layers[38].output[0].save()
                    h39 = model.model.layers[39].output[0].save()
                    h40 = model.model.layers[40].output[0].save()
                    h41 = model.model.layers[41].output[0].save()
                    h42 = model.model.layers[42].output[0].save()
                    h43 = model.model.layers[43].output[0].save()
                    h44 = model.model.layers[44].output[0].save()
                    h45 = model.model.layers[45].output[0].save()
                    h46 = model.model.layers[46].output[0].save()
                    h47 = model.model.layers[47].output[0].save()
                    h48 = model.model.layers[48].output[0].save()
                    h49 = model.model.layers[49].output[0].save()
                    h50 = model.model.layers[50].output[0].save()
                    h51 = model.model.layers[51].output[0].save()
                    h52 = model.model.layers[52].output[0].save()
                    h53 = model.model.layers[53].output[0].save()
                    h54 = model.model.layers[54].output[0].save()
                    h55 = model.model.layers[55].output[0].save()
                    h56 = model.model.layers[56].output[0].save()
                    h57 = model.model.layers[57].output[0].save()
                    h58 = model.model.layers[58].output[0].save()
                    h59 = model.model.layers[59].output[0].save()
                    h60 = model.model.layers[60].output[0].save()
                    h61 = model.model.layers[61].output[0].save()
                    h62 = model.model.layers[62].output[0].save()
                    h63 = model.model.layers[63].output[0].save()
                    h64 = model.model.layers[64].output[0].save()
                    h65 = model.model.layers[65].output[0].save()
                    h66 = model.model.layers[66].output[0].save()
                    h67 = model.model.layers[67].output[0].save()
                    h68 = model.model.layers[68].output[0].save()
                    h69 = model.model.layers[69].output[0].save()
                    h70 = model.model.layers[70].output[0].save()
                    h71 = model.model.layers[71].output[0].save()
                    h72 = model.model.layers[72].output[0].save()
                    h73 = model.model.layers[73].output[0].save()
                    h74 = model.model.layers[74].output[0].save()
                    h75 = model.model.layers[75].output[0].save()
                    h76 = model.model.layers[76].output[0].save()
                    h77 = model.model.layers[77].output[0].save()
                    h78 = model.model.layers[78].output[0].save()
                    h79 = model.model.layers[79].output[0].save()

                success = True
                break
            except Exception as e:
                print(f"  Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2 + random.random() * 5)
        
        if not success:
            print(f"SKIPPING BATCH due to repeated failures.")
            continue

        h0_val = h0.value if hasattr(h0, 'value') else h0 # Check first one
        
        current_batch_size = len(batch)
        layer_proxies = {}
        layer_proxies[0] = h0
        layer_proxies[1] = h1
        layer_proxies[2] = h2
        layer_proxies[3] = h3
        layer_proxies[4] = h4
        layer_proxies[5] = h5
        layer_proxies[6] = h6
        layer_proxies[7] = h7
        layer_proxies[8] = h8
        layer_proxies[9] = h9
        layer_proxies[10] = h10
        layer_proxies[11] = h11
        layer_proxies[12] = h12
        layer_proxies[13] = h13
        layer_proxies[14] = h14
        layer_proxies[15] = h15
        layer_proxies[16] = h16
        layer_proxies[17] = h17
        layer_proxies[18] = h18
        layer_proxies[19] = h19
        layer_proxies[20] = h20
        layer_proxies[21] = h21
        layer_proxies[22] = h22
        layer_proxies[23] = h23
        layer_proxies[24] = h24
        layer_proxies[25] = h25
        layer_proxies[26] = h26
        layer_proxies[27] = h27
        layer_proxies[28] = h28
        layer_proxies[29] = h29
        layer_proxies[30] = h30
        layer_proxies[31] = h31
        layer_proxies[32] = h32
        layer_proxies[33] = h33
        layer_proxies[34] = h34
        layer_proxies[35] = h35
        layer_proxies[36] = h36
        layer_proxies[37] = h37
        layer_proxies[38] = h38
        layer_proxies[39] = h39
        layer_proxies[40] = h40
        layer_proxies[41] = h41
        layer_proxies[42] = h42
        layer_proxies[43] = h43
        layer_proxies[44] = h44
        layer_proxies[45] = h45
        layer_proxies[46] = h46
        layer_proxies[47] = h47
        layer_proxies[48] = h48
        layer_proxies[49] = h49
        layer_proxies[50] = h50
        layer_proxies[51] = h51
        layer_proxies[52] = h52
        layer_proxies[53] = h53
        layer_proxies[54] = h54
        layer_proxies[55] = h55
        layer_proxies[56] = h56
        layer_proxies[57] = h57
        layer_proxies[58] = h58
        layer_proxies[59] = h59
        layer_proxies[60] = h60
        layer_proxies[61] = h61
        layer_proxies[62] = h62
        layer_proxies[63] = h63
        layer_proxies[64] = h64
        layer_proxies[65] = h65
        layer_proxies[66] = h66
        layer_proxies[67] = h67
        layer_proxies[68] = h68
        layer_proxies[69] = h69
        layer_proxies[70] = h70
        layer_proxies[71] = h71
        layer_proxies[72] = h72
        layer_proxies[73] = h73
        layer_proxies[74] = h74
        layer_proxies[75] = h75
        layer_proxies[76] = h76
        layer_proxies[77] = h77
        layer_proxies[78] = h78
        layer_proxies[79] = h79


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
    layer_means = {}
    for layer_idx in layers:
        if not baseline_acts[layer_idx]:
             print(f"WARNING: No acts for layer {layer_idx}")
             continue
        all_acts = torch.stack(baseline_acts[layer_idx], dim=0) 
        layer_means[layer_idx] = all_acts.mean(dim=0)

    # --- 3. Compute Concepts ---
    print(f"Computing {len(CONCEPTS)} concepts...")
    
    concept_chunks = [CONCEPTS[i:i + BATCH_SIZE] for i in range(0, len(CONCEPTS), BATCH_SIZE)]
    
    for batch_idx, batch in enumerate(concept_chunks):
        print(f"Concept Batch {batch_idx+1}/{len(concept_chunks)} ({len(batch)} items)")
        prompts = [f"Tell me about {word}" for word in batch]
        
        # Retry Loop
        max_retries = 5
        success = False
        for attempt in range(max_retries):
            try:
                with model.trace(prompts, remote=True):
                    c0 = model.model.layers[0].output[0].save()
                    c1 = model.model.layers[1].output[0].save()
                    c2 = model.model.layers[2].output[0].save()
                    c3 = model.model.layers[3].output[0].save()
                    c4 = model.model.layers[4].output[0].save()
                    c5 = model.model.layers[5].output[0].save()
                    c6 = model.model.layers[6].output[0].save()
                    c7 = model.model.layers[7].output[0].save()
                    c8 = model.model.layers[8].output[0].save()
                    c9 = model.model.layers[9].output[0].save()
                    c10 = model.model.layers[10].output[0].save()
                    c11 = model.model.layers[11].output[0].save()
                    c12 = model.model.layers[12].output[0].save()
                    c13 = model.model.layers[13].output[0].save()
                    c14 = model.model.layers[14].output[0].save()
                    c15 = model.model.layers[15].output[0].save()
                    c16 = model.model.layers[16].output[0].save()
                    c17 = model.model.layers[17].output[0].save()
                    c18 = model.model.layers[18].output[0].save()
                    c19 = model.model.layers[19].output[0].save()
                    c20 = model.model.layers[20].output[0].save()
                    c21 = model.model.layers[21].output[0].save()
                    c22 = model.model.layers[22].output[0].save()
                    c23 = model.model.layers[23].output[0].save()
                    c24 = model.model.layers[24].output[0].save()
                    c25 = model.model.layers[25].output[0].save()
                    c26 = model.model.layers[26].output[0].save()
                    c27 = model.model.layers[27].output[0].save()
                    c28 = model.model.layers[28].output[0].save()
                    c29 = model.model.layers[29].output[0].save()
                    c30 = model.model.layers[30].output[0].save()
                    c31 = model.model.layers[31].output[0].save()
                    c32 = model.model.layers[32].output[0].save()
                    c33 = model.model.layers[33].output[0].save()
                    c34 = model.model.layers[34].output[0].save()
                    c35 = model.model.layers[35].output[0].save()
                    c36 = model.model.layers[36].output[0].save()
                    c37 = model.model.layers[37].output[0].save()
                    c38 = model.model.layers[38].output[0].save()
                    c39 = model.model.layers[39].output[0].save()
                    c40 = model.model.layers[40].output[0].save()
                    c41 = model.model.layers[41].output[0].save()
                    c42 = model.model.layers[42].output[0].save()
                    c43 = model.model.layers[43].output[0].save()
                    c44 = model.model.layers[44].output[0].save()
                    c45 = model.model.layers[45].output[0].save()
                    c46 = model.model.layers[46].output[0].save()
                    c47 = model.model.layers[47].output[0].save()
                    c48 = model.model.layers[48].output[0].save()
                    c49 = model.model.layers[49].output[0].save()
                    c50 = model.model.layers[50].output[0].save()
                    c51 = model.model.layers[51].output[0].save()
                    c52 = model.model.layers[52].output[0].save()
                    c53 = model.model.layers[53].output[0].save()
                    c54 = model.model.layers[54].output[0].save()
                    c55 = model.model.layers[55].output[0].save()
                    c56 = model.model.layers[56].output[0].save()
                    c57 = model.model.layers[57].output[0].save()
                    c58 = model.model.layers[58].output[0].save()
                    c59 = model.model.layers[59].output[0].save()
                    c60 = model.model.layers[60].output[0].save()
                    c61 = model.model.layers[61].output[0].save()
                    c62 = model.model.layers[62].output[0].save()
                    c63 = model.model.layers[63].output[0].save()
                    c64 = model.model.layers[64].output[0].save()
                    c65 = model.model.layers[65].output[0].save()
                    c66 = model.model.layers[66].output[0].save()
                    c67 = model.model.layers[67].output[0].save()
                    c68 = model.model.layers[68].output[0].save()
                    c69 = model.model.layers[69].output[0].save()
                    c70 = model.model.layers[70].output[0].save()
                    c71 = model.model.layers[71].output[0].save()
                    c72 = model.model.layers[72].output[0].save()
                    c73 = model.model.layers[73].output[0].save()
                    c74 = model.model.layers[74].output[0].save()
                    c75 = model.model.layers[75].output[0].save()
                    c76 = model.model.layers[76].output[0].save()
                    c77 = model.model.layers[77].output[0].save()
                    c78 = model.model.layers[78].output[0].save()
                    c79 = model.model.layers[79].output[0].save()

                success = True
                break
            except Exception as e:
                print(f"  Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2 + random.random() * 5)

        if not success:
            print(f"SKIPPING BATCH due to failures.")
            continue

        concept_proxies = {}
        concept_proxies[0] = c0
        concept_proxies[1] = c1
        concept_proxies[2] = c2
        concept_proxies[3] = c3
        concept_proxies[4] = c4
        concept_proxies[5] = c5
        concept_proxies[6] = c6
        concept_proxies[7] = c7
        concept_proxies[8] = c8
        concept_proxies[9] = c9
        concept_proxies[10] = c10
        concept_proxies[11] = c11
        concept_proxies[12] = c12
        concept_proxies[13] = c13
        concept_proxies[14] = c14
        concept_proxies[15] = c15
        concept_proxies[16] = c16
        concept_proxies[17] = c17
        concept_proxies[18] = c18
        concept_proxies[19] = c19
        concept_proxies[20] = c20
        concept_proxies[21] = c21
        concept_proxies[22] = c22
        concept_proxies[23] = c23
        concept_proxies[24] = c24
        concept_proxies[25] = c25
        concept_proxies[26] = c26
        concept_proxies[27] = c27
        concept_proxies[28] = c28
        concept_proxies[29] = c29
        concept_proxies[30] = c30
        concept_proxies[31] = c31
        concept_proxies[32] = c32
        concept_proxies[33] = c33
        concept_proxies[34] = c34
        concept_proxies[35] = c35
        concept_proxies[36] = c36
        concept_proxies[37] = c37
        concept_proxies[38] = c38
        concept_proxies[39] = c39
        concept_proxies[40] = c40
        concept_proxies[41] = c41
        concept_proxies[42] = c42
        concept_proxies[43] = c43
        concept_proxies[44] = c44
        concept_proxies[45] = c45
        concept_proxies[46] = c46
        concept_proxies[47] = c47
        concept_proxies[48] = c48
        concept_proxies[49] = c49
        concept_proxies[50] = c50
        concept_proxies[51] = c51
        concept_proxies[52] = c52
        concept_proxies[53] = c53
        concept_proxies[54] = c54
        concept_proxies[55] = c55
        concept_proxies[56] = c56
        concept_proxies[57] = c57
        concept_proxies[58] = c58
        concept_proxies[59] = c59
        concept_proxies[60] = c60
        concept_proxies[61] = c61
        concept_proxies[62] = c62
        concept_proxies[63] = c63
        concept_proxies[64] = c64
        concept_proxies[65] = c65
        concept_proxies[66] = c66
        concept_proxies[67] = c67
        concept_proxies[68] = c68
        concept_proxies[69] = c69
        concept_proxies[70] = c70
        concept_proxies[71] = c71
        concept_proxies[72] = c72
        concept_proxies[73] = c73
        concept_proxies[74] = c74
        concept_proxies[75] = c75
        concept_proxies[76] = c76
        concept_proxies[77] = c77
        concept_proxies[78] = c78
        concept_proxies[79] = c79

        
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
