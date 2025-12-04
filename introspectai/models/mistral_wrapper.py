import torch
from transformers import BatchEncoding
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage, SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

class MistralCommonTokenizer:
    def __init__(self, is_tekken=True):
        self.tokenizer = MistralTokenizer.v3(is_tekken=is_tekken)
        self.pad_token_id = 0 # Usually 0 or EOS
        self.eos_token_id = self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        self.bos_token_id = self.tokenizer.instruct_tokenizer.tokenizer.bos_id
        
    def encode(self, text, add_special_tokens=True, return_tensors=None):
        # Basic encoding
        # mistral_common is chat-centric, but we can encode raw text via the underlying tokenizer
        # tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=True, eos=False)
        
        # Note: mistral_common v3 exposes .instruct_tokenizer.tokenizer which is a Tekken/SentencePiece tokenizer
        ids = self.tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=add_special_tokens, eos=False)
        
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids
        
    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                token_ids = token_ids[0]
            token_ids = token_ids.tolist()
            
        return self.tokenizer.instruct_tokenizer.tokenizer.decode(token_ids)
        
    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None):
        # Convert dict messages to mistral_common objects
        mistral_msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "user":
                mistral_msgs.append(UserMessage(content=content))
            elif role == "assistant":
                mistral_msgs.append(AssistantMessage(content=content))
            elif role == "system":
                mistral_msgs.append(SystemMessage(content=content))
                
        req = ChatCompletionRequest(messages=mistral_msgs)
        tokenized = self.tokenizer.encode_chat_completion(req)
        ids = tokenized.tokens
        
        # mistral_common adds generation prompt automatically if it's a user message at the end?
        # Actually encode_chat_completion prepares input for the model.
        
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids
        
    def __call__(self, text, return_tensors=None, **kwargs):
        # Mimic tokenizer(text) call
        return BatchEncoding({"input_ids": self.encode(text, return_tensors=return_tensors)})
