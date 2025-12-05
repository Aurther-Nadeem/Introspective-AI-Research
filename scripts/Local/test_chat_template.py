from nnsight import LanguageModel, CONFIG

# Set API Key
CONFIG.set_default_api_key("3ab78817-4682-4d8e-ac5d-73d68dc308ce")

def test_generate():
    print("Initializing model...")
    model = LanguageModel("meta-llama/Llama-3.1-405B-Instruct")
    
    prompt = "Hello, how are you?"
    
    print("Running generation with remote=True...")
    try:
        with model.generate(prompt, max_new_tokens=10, remote=True) as gen:
            # Try the user's suggested pattern
            saved = model.generator.output.save()
            
        print("Generation complete.")
        print(f"Saved type: {type(saved)}")
        
        print(f"Saved type: {type(saved)}")
        
        val = saved.value if hasattr(saved, 'value') else saved
        print(f"Value type: {type(val)}")
        
        if hasattr(val, 'shape'):
            print(f"Value shape: {val.shape}")
        
        # Decode
        # val might be [batch, seq_len]
        if isinstance(val, list):
             decoded = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in val]
        else:
             decoded = model.tokenizer.batch_decode(val, skip_special_tokens=True)
             
        print(f"Decoded: {decoded}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generate()
