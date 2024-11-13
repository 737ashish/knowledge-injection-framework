import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, StoppingCriteria

class CustomStoppingCriteria(StoppingCriteria):
    """
    Allows to use list of strings which can be used as stop tokens in generation.
    Usage: model.generate(..., stopping_criteria=StoppingCriteriaList([".", "?", "!"], prompt, tokenizer))
    """
    def __init__(self, stop_sequences, prompt, tokenizer):
        self.stop_sequences = stop_sequences
        self.prompt = prompt
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        """Checks if generation contains any of the strings in stop_sequences."""
        generation = self.tokenizer.decode(input_ids[0])[len(self.prompt):]
        if any([stop_sequence in generation for stop_sequence in self.stop_sequences]):
            return True # Generation stops
        return False # Generation continues

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

class CustomLogitsProcessor(LogitsProcessor):
    """
    Allow custom processing of logits before sampling next token.
    Usage: LogitsProcessorList([CustomLogitsProcessor(args)]))
    """
    def __init__(self):
        # self.arg = arg
        pass

    def __call__(self, input_ids, scores):
        # scores = logits
        return scores

def load_model(model_name):     
    if "mistral" in model_name.lower():
        hf_token = "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"        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     device_map="auto", 
                                                     load_in_4bit=True,
                                                     token=hf_token,
                                                     bnb_4bit_compute_dtype=torch.float16)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif "c4ai-command-r-plus" in model_name.lower():
        hf_token = "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"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token = tokenizer.bos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     device_map="auto", 
                                                     load_in_4bit=True, 
                                                     token=hf_token, 
                                                     bnb_4bit_compute_dtype=torch.float16
                                                     )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif "qwen" in model_name.lower():
        hf_token = "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"        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        device_map="auto", 
                                                        load_in_4bit=True,
                                                        torch_dtype=torch.float32,
                                                        token=hf_token,
                                                        bnb_4bit_compute_dtype=torch.float16)

    return tokenizer, model

def generate(prompt, tokenizer, model, max_new_tokens=30, stop_sequences=None, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if stop_sequences:
        stopping_criteria = CustomStoppingCriteria(stop_sequences, prompt, tokenizer)
    else:
        stopping_criteria = None
    outputs = model.generate(**inputs, 
                            max_new_tokens=max_new_tokens, 
                            do_sample=False, 
                            stopping_criteria=stopping_criteria,
                            **kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_top_k_tokens(logits, top_k, tokenizer):
    """Returns list with tuples of top_k tokens with probabilities given a tensor of logits."""
    top_k_logits, top_k_token_indices = torch.topk(logits, k=top_k)
    probas = torch.softmax(top_k_logits, dim=-1)
    tokens = [tokenizer.decode(x) for x in top_k_token_indices]
    return list(zip(tokens, probas.tolist()))

def print_next_token_probas(prompt, tokenizer, model, top_k=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :][0]
    top_k_tokens = get_top_k_tokens(logits, top_k, tokenizer)
    for token, proba in top_k_tokens:
        print(f"{proba:.4f} - {repr(token)}")

def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# number of parameters in billions
model_size_dict = {
    "gpt2-medium": 0.355,
    "gpt2-large": 0.774,
    "gpt2-xl": 1.5,
    "gpt-j-6b": 6,
    "llama-2-7b": 7,
    "llama-2-13b": 13,
    "llama-2-70b": 70
}
    