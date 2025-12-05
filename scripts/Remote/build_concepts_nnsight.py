import torch
import json
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
    model_name = "meta-llama/Llama-3.1-405B-Instruct"
    print(f"Initializing NNsight model: {model_name}")
    model = LanguageModel(model_name)
    store = NNsightConceptStore(model_name=model_name)
    
    layers = list(range(126))
    print(f"Targeting all 126 layers.")
    
    # 1. Get Baseline Activations
    print(f"Computing baseline ({len(BASELINE_WORDS)} words)...")
    baseline_acts = {l: [] for l in layers}
    
    for i, word in enumerate(BASELINE_WORDS):
        if i % 10 == 0:
            print(f"Baseline word {i+1}/{len(BASELINE_WORDS)}: {word}")
            
        prompt = f"Tell me about {word}"
        
        # Unrolled trace block
        with model.trace(prompt, remote=True):
            h0 = model.model.layers[0].output[0][-1, :].save()
            h1 = model.model.layers[1].output[0][-1, :].save()
            h2 = model.model.layers[2].output[0][-1, :].save()
            h3 = model.model.layers[3].output[0][-1, :].save()
            h4 = model.model.layers[4].output[0][-1, :].save()
            h5 = model.model.layers[5].output[0][-1, :].save()
            h6 = model.model.layers[6].output[0][-1, :].save()
            h7 = model.model.layers[7].output[0][-1, :].save()
            h8 = model.model.layers[8].output[0][-1, :].save()
            h9 = model.model.layers[9].output[0][-1, :].save()
            h10 = model.model.layers[10].output[0][-1, :].save()
            h11 = model.model.layers[11].output[0][-1, :].save()
            h12 = model.model.layers[12].output[0][-1, :].save()
            h13 = model.model.layers[13].output[0][-1, :].save()
            h14 = model.model.layers[14].output[0][-1, :].save()
            h15 = model.model.layers[15].output[0][-1, :].save()
            h16 = model.model.layers[16].output[0][-1, :].save()
            h17 = model.model.layers[17].output[0][-1, :].save()
            h18 = model.model.layers[18].output[0][-1, :].save()
            h19 = model.model.layers[19].output[0][-1, :].save()
            h20 = model.model.layers[20].output[0][-1, :].save()
            h21 = model.model.layers[21].output[0][-1, :].save()
            h22 = model.model.layers[22].output[0][-1, :].save()
            h23 = model.model.layers[23].output[0][-1, :].save()
            h24 = model.model.layers[24].output[0][-1, :].save()
            h25 = model.model.layers[25].output[0][-1, :].save()
            h26 = model.model.layers[26].output[0][-1, :].save()
            h27 = model.model.layers[27].output[0][-1, :].save()
            h28 = model.model.layers[28].output[0][-1, :].save()
            h29 = model.model.layers[29].output[0][-1, :].save()
            h30 = model.model.layers[30].output[0][-1, :].save()
            h31 = model.model.layers[31].output[0][-1, :].save()
            h32 = model.model.layers[32].output[0][-1, :].save()
            h33 = model.model.layers[33].output[0][-1, :].save()
            h34 = model.model.layers[34].output[0][-1, :].save()
            h35 = model.model.layers[35].output[0][-1, :].save()
            h36 = model.model.layers[36].output[0][-1, :].save()
            h37 = model.model.layers[37].output[0][-1, :].save()
            h38 = model.model.layers[38].output[0][-1, :].save()
            h39 = model.model.layers[39].output[0][-1, :].save()
            h40 = model.model.layers[40].output[0][-1, :].save()
            h41 = model.model.layers[41].output[0][-1, :].save()
            h42 = model.model.layers[42].output[0][-1, :].save()
            h43 = model.model.layers[43].output[0][-1, :].save()
            h44 = model.model.layers[44].output[0][-1, :].save()
            h45 = model.model.layers[45].output[0][-1, :].save()
            h46 = model.model.layers[46].output[0][-1, :].save()
            h47 = model.model.layers[47].output[0][-1, :].save()
            h48 = model.model.layers[48].output[0][-1, :].save()
            h49 = model.model.layers[49].output[0][-1, :].save()
            h50 = model.model.layers[50].output[0][-1, :].save()
            h51 = model.model.layers[51].output[0][-1, :].save()
            h52 = model.model.layers[52].output[0][-1, :].save()
            h53 = model.model.layers[53].output[0][-1, :].save()
            h54 = model.model.layers[54].output[0][-1, :].save()
            h55 = model.model.layers[55].output[0][-1, :].save()
            h56 = model.model.layers[56].output[0][-1, :].save()
            h57 = model.model.layers[57].output[0][-1, :].save()
            h58 = model.model.layers[58].output[0][-1, :].save()
            h59 = model.model.layers[59].output[0][-1, :].save()
            h60 = model.model.layers[60].output[0][-1, :].save()
            h61 = model.model.layers[61].output[0][-1, :].save()
            h62 = model.model.layers[62].output[0][-1, :].save()
            h63 = model.model.layers[63].output[0][-1, :].save()
            h64 = model.model.layers[64].output[0][-1, :].save()
            h65 = model.model.layers[65].output[0][-1, :].save()
            h66 = model.model.layers[66].output[0][-1, :].save()
            h67 = model.model.layers[67].output[0][-1, :].save()
            h68 = model.model.layers[68].output[0][-1, :].save()
            h69 = model.model.layers[69].output[0][-1, :].save()
            h70 = model.model.layers[70].output[0][-1, :].save()
            h71 = model.model.layers[71].output[0][-1, :].save()
            h72 = model.model.layers[72].output[0][-1, :].save()
            h73 = model.model.layers[73].output[0][-1, :].save()
            h74 = model.model.layers[74].output[0][-1, :].save()
            h75 = model.model.layers[75].output[0][-1, :].save()
            h76 = model.model.layers[76].output[0][-1, :].save()
            h77 = model.model.layers[77].output[0][-1, :].save()
            h78 = model.model.layers[78].output[0][-1, :].save()
            h79 = model.model.layers[79].output[0][-1, :].save()
            h80 = model.model.layers[80].output[0][-1, :].save()
            h81 = model.model.layers[81].output[0][-1, :].save()
            h82 = model.model.layers[82].output[0][-1, :].save()
            h83 = model.model.layers[83].output[0][-1, :].save()
            h84 = model.model.layers[84].output[0][-1, :].save()
            h85 = model.model.layers[85].output[0][-1, :].save()
            h86 = model.model.layers[86].output[0][-1, :].save()
            h87 = model.model.layers[87].output[0][-1, :].save()
            h88 = model.model.layers[88].output[0][-1, :].save()
            h89 = model.model.layers[89].output[0][-1, :].save()
            h90 = model.model.layers[90].output[0][-1, :].save()
            h91 = model.model.layers[91].output[0][-1, :].save()
            h92 = model.model.layers[92].output[0][-1, :].save()
            h93 = model.model.layers[93].output[0][-1, :].save()
            h94 = model.model.layers[94].output[0][-1, :].save()
            h95 = model.model.layers[95].output[0][-1, :].save()
            h96 = model.model.layers[96].output[0][-1, :].save()
            h97 = model.model.layers[97].output[0][-1, :].save()
            h98 = model.model.layers[98].output[0][-1, :].save()
            h99 = model.model.layers[99].output[0][-1, :].save()
            h100 = model.model.layers[100].output[0][-1, :].save()
            h101 = model.model.layers[101].output[0][-1, :].save()
            h102 = model.model.layers[102].output[0][-1, :].save()
            h103 = model.model.layers[103].output[0][-1, :].save()
            h104 = model.model.layers[104].output[0][-1, :].save()
            h105 = model.model.layers[105].output[0][-1, :].save()
            h106 = model.model.layers[106].output[0][-1, :].save()
            h107 = model.model.layers[107].output[0][-1, :].save()
            h108 = model.model.layers[108].output[0][-1, :].save()
            h109 = model.model.layers[109].output[0][-1, :].save()
            h110 = model.model.layers[110].output[0][-1, :].save()
            h111 = model.model.layers[111].output[0][-1, :].save()
            h112 = model.model.layers[112].output[0][-1, :].save()
            h113 = model.model.layers[113].output[0][-1, :].save()
            h114 = model.model.layers[114].output[0][-1, :].save()
            h115 = model.model.layers[115].output[0][-1, :].save()
            h116 = model.model.layers[116].output[0][-1, :].save()
            h117 = model.model.layers[117].output[0][-1, :].save()
            h118 = model.model.layers[118].output[0][-1, :].save()
            h119 = model.model.layers[119].output[0][-1, :].save()
            h120 = model.model.layers[120].output[0][-1, :].save()
            h121 = model.model.layers[121].output[0][-1, :].save()
            h122 = model.model.layers[122].output[0][-1, :].save()
            h123 = model.model.layers[123].output[0][-1, :].save()
            h124 = model.model.layers[124].output[0][-1, :].save()
            h125 = model.model.layers[125].output[0][-1, :].save()

        # Collect outputs
        layer_outputs = {}
        layer_outputs[0] = h0
        layer_outputs[1] = h1
        layer_outputs[2] = h2
        layer_outputs[3] = h3
        layer_outputs[4] = h4
        layer_outputs[5] = h5
        layer_outputs[6] = h6
        layer_outputs[7] = h7
        layer_outputs[8] = h8
        layer_outputs[9] = h9
        layer_outputs[10] = h10
        layer_outputs[11] = h11
        layer_outputs[12] = h12
        layer_outputs[13] = h13
        layer_outputs[14] = h14
        layer_outputs[15] = h15
        layer_outputs[16] = h16
        layer_outputs[17] = h17
        layer_outputs[18] = h18
        layer_outputs[19] = h19
        layer_outputs[20] = h20
        layer_outputs[21] = h21
        layer_outputs[22] = h22
        layer_outputs[23] = h23
        layer_outputs[24] = h24
        layer_outputs[25] = h25
        layer_outputs[26] = h26
        layer_outputs[27] = h27
        layer_outputs[28] = h28
        layer_outputs[29] = h29
        layer_outputs[30] = h30
        layer_outputs[31] = h31
        layer_outputs[32] = h32
        layer_outputs[33] = h33
        layer_outputs[34] = h34
        layer_outputs[35] = h35
        layer_outputs[36] = h36
        layer_outputs[37] = h37
        layer_outputs[38] = h38
        layer_outputs[39] = h39
        layer_outputs[40] = h40
        layer_outputs[41] = h41
        layer_outputs[42] = h42
        layer_outputs[43] = h43
        layer_outputs[44] = h44
        layer_outputs[45] = h45
        layer_outputs[46] = h46
        layer_outputs[47] = h47
        layer_outputs[48] = h48
        layer_outputs[49] = h49
        layer_outputs[50] = h50
        layer_outputs[51] = h51
        layer_outputs[52] = h52
        layer_outputs[53] = h53
        layer_outputs[54] = h54
        layer_outputs[55] = h55
        layer_outputs[56] = h56
        layer_outputs[57] = h57
        layer_outputs[58] = h58
        layer_outputs[59] = h59
        layer_outputs[60] = h60
        layer_outputs[61] = h61
        layer_outputs[62] = h62
        layer_outputs[63] = h63
        layer_outputs[64] = h64
        layer_outputs[65] = h65
        layer_outputs[66] = h66
        layer_outputs[67] = h67
        layer_outputs[68] = h68
        layer_outputs[69] = h69
        layer_outputs[70] = h70
        layer_outputs[71] = h71
        layer_outputs[72] = h72
        layer_outputs[73] = h73
        layer_outputs[74] = h74
        layer_outputs[75] = h75
        layer_outputs[76] = h76
        layer_outputs[77] = h77
        layer_outputs[78] = h78
        layer_outputs[79] = h79
        layer_outputs[80] = h80
        layer_outputs[81] = h81
        layer_outputs[82] = h82
        layer_outputs[83] = h83
        layer_outputs[84] = h84
        layer_outputs[85] = h85
        layer_outputs[86] = h86
        layer_outputs[87] = h87
        layer_outputs[88] = h88
        layer_outputs[89] = h89
        layer_outputs[90] = h90
        layer_outputs[91] = h91
        layer_outputs[92] = h92
        layer_outputs[93] = h93
        layer_outputs[94] = h94
        layer_outputs[95] = h95
        layer_outputs[96] = h96
        layer_outputs[97] = h97
        layer_outputs[98] = h98
        layer_outputs[99] = h99
        layer_outputs[100] = h100
        layer_outputs[101] = h101
        layer_outputs[102] = h102
        layer_outputs[103] = h103
        layer_outputs[104] = h104
        layer_outputs[105] = h105
        layer_outputs[106] = h106
        layer_outputs[107] = h107
        layer_outputs[108] = h108
        layer_outputs[109] = h109
        layer_outputs[110] = h110
        layer_outputs[111] = h111
        layer_outputs[112] = h112
        layer_outputs[113] = h113
        layer_outputs[114] = h114
        layer_outputs[115] = h115
        layer_outputs[116] = h116
        layer_outputs[117] = h117
        layer_outputs[118] = h118
        layer_outputs[119] = h119
        layer_outputs[120] = h120
        layer_outputs[121] = h121
        layer_outputs[122] = h122
        layer_outputs[123] = h123
        layer_outputs[124] = h124
        layer_outputs[125] = h125

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
            c0 = model.model.layers[0].output[0][-1, :].save()
            c1 = model.model.layers[1].output[0][-1, :].save()
            c2 = model.model.layers[2].output[0][-1, :].save()
            c3 = model.model.layers[3].output[0][-1, :].save()
            c4 = model.model.layers[4].output[0][-1, :].save()
            c5 = model.model.layers[5].output[0][-1, :].save()
            c6 = model.model.layers[6].output[0][-1, :].save()
            c7 = model.model.layers[7].output[0][-1, :].save()
            c8 = model.model.layers[8].output[0][-1, :].save()
            c9 = model.model.layers[9].output[0][-1, :].save()
            c10 = model.model.layers[10].output[0][-1, :].save()
            c11 = model.model.layers[11].output[0][-1, :].save()
            c12 = model.model.layers[12].output[0][-1, :].save()
            c13 = model.model.layers[13].output[0][-1, :].save()
            c14 = model.model.layers[14].output[0][-1, :].save()
            c15 = model.model.layers[15].output[0][-1, :].save()
            c16 = model.model.layers[16].output[0][-1, :].save()
            c17 = model.model.layers[17].output[0][-1, :].save()
            c18 = model.model.layers[18].output[0][-1, :].save()
            c19 = model.model.layers[19].output[0][-1, :].save()
            c20 = model.model.layers[20].output[0][-1, :].save()
            c21 = model.model.layers[21].output[0][-1, :].save()
            c22 = model.model.layers[22].output[0][-1, :].save()
            c23 = model.model.layers[23].output[0][-1, :].save()
            c24 = model.model.layers[24].output[0][-1, :].save()
            c25 = model.model.layers[25].output[0][-1, :].save()
            c26 = model.model.layers[26].output[0][-1, :].save()
            c27 = model.model.layers[27].output[0][-1, :].save()
            c28 = model.model.layers[28].output[0][-1, :].save()
            c29 = model.model.layers[29].output[0][-1, :].save()
            c30 = model.model.layers[30].output[0][-1, :].save()
            c31 = model.model.layers[31].output[0][-1, :].save()
            c32 = model.model.layers[32].output[0][-1, :].save()
            c33 = model.model.layers[33].output[0][-1, :].save()
            c34 = model.model.layers[34].output[0][-1, :].save()
            c35 = model.model.layers[35].output[0][-1, :].save()
            c36 = model.model.layers[36].output[0][-1, :].save()
            c37 = model.model.layers[37].output[0][-1, :].save()
            c38 = model.model.layers[38].output[0][-1, :].save()
            c39 = model.model.layers[39].output[0][-1, :].save()
            c40 = model.model.layers[40].output[0][-1, :].save()
            c41 = model.model.layers[41].output[0][-1, :].save()
            c42 = model.model.layers[42].output[0][-1, :].save()
            c43 = model.model.layers[43].output[0][-1, :].save()
            c44 = model.model.layers[44].output[0][-1, :].save()
            c45 = model.model.layers[45].output[0][-1, :].save()
            c46 = model.model.layers[46].output[0][-1, :].save()
            c47 = model.model.layers[47].output[0][-1, :].save()
            c48 = model.model.layers[48].output[0][-1, :].save()
            c49 = model.model.layers[49].output[0][-1, :].save()
            c50 = model.model.layers[50].output[0][-1, :].save()
            c51 = model.model.layers[51].output[0][-1, :].save()
            c52 = model.model.layers[52].output[0][-1, :].save()
            c53 = model.model.layers[53].output[0][-1, :].save()
            c54 = model.model.layers[54].output[0][-1, :].save()
            c55 = model.model.layers[55].output[0][-1, :].save()
            c56 = model.model.layers[56].output[0][-1, :].save()
            c57 = model.model.layers[57].output[0][-1, :].save()
            c58 = model.model.layers[58].output[0][-1, :].save()
            c59 = model.model.layers[59].output[0][-1, :].save()
            c60 = model.model.layers[60].output[0][-1, :].save()
            c61 = model.model.layers[61].output[0][-1, :].save()
            c62 = model.model.layers[62].output[0][-1, :].save()
            c63 = model.model.layers[63].output[0][-1, :].save()
            c64 = model.model.layers[64].output[0][-1, :].save()
            c65 = model.model.layers[65].output[0][-1, :].save()
            c66 = model.model.layers[66].output[0][-1, :].save()
            c67 = model.model.layers[67].output[0][-1, :].save()
            c68 = model.model.layers[68].output[0][-1, :].save()
            c69 = model.model.layers[69].output[0][-1, :].save()
            c70 = model.model.layers[70].output[0][-1, :].save()
            c71 = model.model.layers[71].output[0][-1, :].save()
            c72 = model.model.layers[72].output[0][-1, :].save()
            c73 = model.model.layers[73].output[0][-1, :].save()
            c74 = model.model.layers[74].output[0][-1, :].save()
            c75 = model.model.layers[75].output[0][-1, :].save()
            c76 = model.model.layers[76].output[0][-1, :].save()
            c77 = model.model.layers[77].output[0][-1, :].save()
            c78 = model.model.layers[78].output[0][-1, :].save()
            c79 = model.model.layers[79].output[0][-1, :].save()
            c80 = model.model.layers[80].output[0][-1, :].save()
            c81 = model.model.layers[81].output[0][-1, :].save()
            c82 = model.model.layers[82].output[0][-1, :].save()
            c83 = model.model.layers[83].output[0][-1, :].save()
            c84 = model.model.layers[84].output[0][-1, :].save()
            c85 = model.model.layers[85].output[0][-1, :].save()
            c86 = model.model.layers[86].output[0][-1, :].save()
            c87 = model.model.layers[87].output[0][-1, :].save()
            c88 = model.model.layers[88].output[0][-1, :].save()
            c89 = model.model.layers[89].output[0][-1, :].save()
            c90 = model.model.layers[90].output[0][-1, :].save()
            c91 = model.model.layers[91].output[0][-1, :].save()
            c92 = model.model.layers[92].output[0][-1, :].save()
            c93 = model.model.layers[93].output[0][-1, :].save()
            c94 = model.model.layers[94].output[0][-1, :].save()
            c95 = model.model.layers[95].output[0][-1, :].save()
            c96 = model.model.layers[96].output[0][-1, :].save()
            c97 = model.model.layers[97].output[0][-1, :].save()
            c98 = model.model.layers[98].output[0][-1, :].save()
            c99 = model.model.layers[99].output[0][-1, :].save()
            c100 = model.model.layers[100].output[0][-1, :].save()
            c101 = model.model.layers[101].output[0][-1, :].save()
            c102 = model.model.layers[102].output[0][-1, :].save()
            c103 = model.model.layers[103].output[0][-1, :].save()
            c104 = model.model.layers[104].output[0][-1, :].save()
            c105 = model.model.layers[105].output[0][-1, :].save()
            c106 = model.model.layers[106].output[0][-1, :].save()
            c107 = model.model.layers[107].output[0][-1, :].save()
            c108 = model.model.layers[108].output[0][-1, :].save()
            c109 = model.model.layers[109].output[0][-1, :].save()
            c110 = model.model.layers[110].output[0][-1, :].save()
            c111 = model.model.layers[111].output[0][-1, :].save()
            c112 = model.model.layers[112].output[0][-1, :].save()
            c113 = model.model.layers[113].output[0][-1, :].save()
            c114 = model.model.layers[114].output[0][-1, :].save()
            c115 = model.model.layers[115].output[0][-1, :].save()
            c116 = model.model.layers[116].output[0][-1, :].save()
            c117 = model.model.layers[117].output[0][-1, :].save()
            c118 = model.model.layers[118].output[0][-1, :].save()
            c119 = model.model.layers[119].output[0][-1, :].save()
            c120 = model.model.layers[120].output[0][-1, :].save()
            c121 = model.model.layers[121].output[0][-1, :].save()
            c122 = model.model.layers[122].output[0][-1, :].save()
            c123 = model.model.layers[123].output[0][-1, :].save()
            c124 = model.model.layers[124].output[0][-1, :].save()
            c125 = model.model.layers[125].output[0][-1, :].save()

        # Collect outputs
        concept_acts_proxies = {}
        concept_acts_proxies[0] = c0
        concept_acts_proxies[1] = c1
        concept_acts_proxies[2] = c2
        concept_acts_proxies[3] = c3
        concept_acts_proxies[4] = c4
        concept_acts_proxies[5] = c5
        concept_acts_proxies[6] = c6
        concept_acts_proxies[7] = c7
        concept_acts_proxies[8] = c8
        concept_acts_proxies[9] = c9
        concept_acts_proxies[10] = c10
        concept_acts_proxies[11] = c11
        concept_acts_proxies[12] = c12
        concept_acts_proxies[13] = c13
        concept_acts_proxies[14] = c14
        concept_acts_proxies[15] = c15
        concept_acts_proxies[16] = c16
        concept_acts_proxies[17] = c17
        concept_acts_proxies[18] = c18
        concept_acts_proxies[19] = c19
        concept_acts_proxies[20] = c20
        concept_acts_proxies[21] = c21
        concept_acts_proxies[22] = c22
        concept_acts_proxies[23] = c23
        concept_acts_proxies[24] = c24
        concept_acts_proxies[25] = c25
        concept_acts_proxies[26] = c26
        concept_acts_proxies[27] = c27
        concept_acts_proxies[28] = c28
        concept_acts_proxies[29] = c29
        concept_acts_proxies[30] = c30
        concept_acts_proxies[31] = c31
        concept_acts_proxies[32] = c32
        concept_acts_proxies[33] = c33
        concept_acts_proxies[34] = c34
        concept_acts_proxies[35] = c35
        concept_acts_proxies[36] = c36
        concept_acts_proxies[37] = c37
        concept_acts_proxies[38] = c38
        concept_acts_proxies[39] = c39
        concept_acts_proxies[40] = c40
        concept_acts_proxies[41] = c41
        concept_acts_proxies[42] = c42
        concept_acts_proxies[43] = c43
        concept_acts_proxies[44] = c44
        concept_acts_proxies[45] = c45
        concept_acts_proxies[46] = c46
        concept_acts_proxies[47] = c47
        concept_acts_proxies[48] = c48
        concept_acts_proxies[49] = c49
        concept_acts_proxies[50] = c50
        concept_acts_proxies[51] = c51
        concept_acts_proxies[52] = c52
        concept_acts_proxies[53] = c53
        concept_acts_proxies[54] = c54
        concept_acts_proxies[55] = c55
        concept_acts_proxies[56] = c56
        concept_acts_proxies[57] = c57
        concept_acts_proxies[58] = c58
        concept_acts_proxies[59] = c59
        concept_acts_proxies[60] = c60
        concept_acts_proxies[61] = c61
        concept_acts_proxies[62] = c62
        concept_acts_proxies[63] = c63
        concept_acts_proxies[64] = c64
        concept_acts_proxies[65] = c65
        concept_acts_proxies[66] = c66
        concept_acts_proxies[67] = c67
        concept_acts_proxies[68] = c68
        concept_acts_proxies[69] = c69
        concept_acts_proxies[70] = c70
        concept_acts_proxies[71] = c71
        concept_acts_proxies[72] = c72
        concept_acts_proxies[73] = c73
        concept_acts_proxies[74] = c74
        concept_acts_proxies[75] = c75
        concept_acts_proxies[76] = c76
        concept_acts_proxies[77] = c77
        concept_acts_proxies[78] = c78
        concept_acts_proxies[79] = c79
        concept_acts_proxies[80] = c80
        concept_acts_proxies[81] = c81
        concept_acts_proxies[82] = c82
        concept_acts_proxies[83] = c83
        concept_acts_proxies[84] = c84
        concept_acts_proxies[85] = c85
        concept_acts_proxies[86] = c86
        concept_acts_proxies[87] = c87
        concept_acts_proxies[88] = c88
        concept_acts_proxies[89] = c89
        concept_acts_proxies[90] = c90
        concept_acts_proxies[91] = c91
        concept_acts_proxies[92] = c92
        concept_acts_proxies[93] = c93
        concept_acts_proxies[94] = c94
        concept_acts_proxies[95] = c95
        concept_acts_proxies[96] = c96
        concept_acts_proxies[97] = c97
        concept_acts_proxies[98] = c98
        concept_acts_proxies[99] = c99
        concept_acts_proxies[100] = c100
        concept_acts_proxies[101] = c101
        concept_acts_proxies[102] = c102
        concept_acts_proxies[103] = c103
        concept_acts_proxies[104] = c104
        concept_acts_proxies[105] = c105
        concept_acts_proxies[106] = c106
        concept_acts_proxies[107] = c107
        concept_acts_proxies[108] = c108
        concept_acts_proxies[109] = c109
        concept_acts_proxies[110] = c110
        concept_acts_proxies[111] = c111
        concept_acts_proxies[112] = c112
        concept_acts_proxies[113] = c113
        concept_acts_proxies[114] = c114
        concept_acts_proxies[115] = c115
        concept_acts_proxies[116] = c116
        concept_acts_proxies[117] = c117
        concept_acts_proxies[118] = c118
        concept_acts_proxies[119] = c119
        concept_acts_proxies[120] = c120
        concept_acts_proxies[121] = c121
        concept_acts_proxies[122] = c122
        concept_acts_proxies[123] = c123
        concept_acts_proxies[124] = c124
        concept_acts_proxies[125] = c125

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
