from transformers import AutoTokenizer, LlamaTokenizer

model_name = "mistralai/Ministral-3-8B-Reasoning-2512"

print(f"Testing tokenizer loading for {model_name}...")

print("\n--- Attempt 1: AutoTokenizer (use_fast=False) ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Attempt 2: AutoTokenizer (use_fast=True) ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Attempt 3: LlamaTokenizer ---")
try:
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Attempt 4: mistral_common ---")
try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    # Ministral-3 uses Tekken
    tokenizer = MistralTokenizer.v3(is_tekken=True)
    print("Success! Loaded MistralTokenizer v3 (Tekken)")
    
    # Test encoding
    ids = tokenizer.encode_chat_completion(
        from_mistral_common_request(
            ChatCompletionRequest(messages=[UserMessage(content="Hello")])
        )
    ).tokens
    print(f"Encoded 'Hello': {ids}")
except Exception as e:
    print(f"Failed: {e}")
    # Try importing the request objects if needed
    try:
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        tokenizer = MistralTokenizer.v3(is_tekken=True)
        req = ChatCompletionRequest(messages=[UserMessage(content="Hello")])
        ids = tokenizer.encode_chat_completion(req).tokens
        print(f"Encoded 'Hello' (retry): {ids}")
    except Exception as e2:
        print(f"Failed retry: {e2}")
