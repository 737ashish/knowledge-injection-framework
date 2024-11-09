import nltk
import numpy as np
import torch

def ppl(model, tokenizer, text):
    """
    Computes the perplexity of a text for a reference model.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    log_probs = torch.log_softmax(logits, dim=2)
    # get log probs for token ids in text
    log_probs = torch.gather(log_probs[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])
    return torch.exp(-1 / log_probs.size()[1] * log_probs.sum()).item()

def ppl_generation(model, tokenizer, prompt, generation):
    """
    Computes the perplexity of a generated text given a prompt for a reference model.
    """
    prompt_inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    gen_inputs = tokenizer(generation, return_tensors="pt")["input_ids"]
    inputs = torch.cat((prompt_inputs, gen_inputs), dim=1).to(model.device)
    with torch.no_grad():
        logits = model(inputs).logits
    log_probs = torch.log_softmax(logits, dim=2)
    # get log probs for token ids in generation
    log_probs = torch.gather(log_probs[:, prompt_inputs.size(1)-1:-1, :], 2, inputs[:, prompt_inputs.size(1):, None])
    return torch.exp(-1 / log_probs.size()[1] * log_probs.sum()).item()

def compute_n_gram_entropy(sentence, ns=None, weights=None, normalize=True):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2/3, 4/3]
    entropy_list = []
    for n in ns:
        ngrams = nltk.ngrams(nltk.word_tokenize(sentence), n)
        fdist = nltk.FreqDist(ngrams)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()
        entropy = np.sum(-freqs * np.log2(freqs))
        if normalize and entropy > 0:
            entropy = entropy / np.log2(len(freqs))
        entropy_list.append(entropy)
    entropy_list = np.array(entropy_list) * np.array(weights)
    return np.mean(entropy_list)

def seq_rep_n(sentence, ns=[2, 3, 4]):
    scores = []
    for n in ns:
        ngrams = nltk.ngrams(nltk.word_tokenize(sentence), n)
        fdist = nltk.FreqDist(ngrams)
        scores.append(len(fdist) / np.sum(list(fdist.values())) if len(fdist) > 0 else 0)
    return np.mean(scores)
