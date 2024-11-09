import numpy as np
import torch

def extract_triples(text, ie_tokenizer, ie_model):
    """
    Returns list of dicts which are triples found in text.
    """
    gen_kwargs = dict(
        max_length = 256,
        length_penalty = 0,
        num_beams = 5,
        num_return_sequences = 3,
    )
    inputs = ie_tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors="pt").to(ie_model.device)
    generated_tokens = ie_model.generate(**inputs, **gen_kwargs)
    decoded_preds = ie_tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # extract triples
    l = []
    for sentence in decoded_preds:
        l.append(format_rebel_output(sentence))
    return [x for sublist in l for x in sublist] # flatten list


def extract_triples_batch(texts, ie_tokenizer, ie_model):
    """
    'texts' is a list of strings.
    Returns list with lists of triples for each text in texts.
    """
    batch_size = len(texts)

    gen_kwargs = dict(
        max_length = 256,
        length_penalty = 0,
        num_beams = 5,
        num_return_sequences = 3,
    )
    inputs = ie_tokenizer(texts, max_length=256, padding=True, truncation=True, return_tensors="pt").to(ie_model.device)
    generated_tokens = ie_model.generate(**inputs, **gen_kwargs)
    decoded_preds = ie_tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    decoded_preds = np.reshape(decoded_preds, (batch_size, gen_kwargs["num_return_sequences"]))

    triples_batch = []
    for pred in decoded_preds:
        # extract triples
        l = []
        for sentence in pred:
            l.append(format_rebel_output(sentence))
        triples_batch.append([x for sublist in l for x in sublist])
    return triples_batch


def format_rebel_output(text):
    """
    Function to format REBEL model sequence output to triples formated as dict.
    """
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append({"h": subject.strip(), "r": relation.strip(),"t": object_.strip()})
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append({"h": subject.strip(), "r": relation.strip(),"t": object_.strip()})
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append({"h": subject.strip(), "r": relation.strip(),"t": object_.strip()})
    return triplets