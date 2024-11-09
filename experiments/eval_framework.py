import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import transformers

from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

sys.path.extend(["../utils", "../evaluate"])
from metrics import compute_n_gram_entropy, seq_rep_n
from transformer_utils import mean_pooling, model_size_dict
from wikidata import load_wikidata_json

os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable tokenizer warning

# set up logging
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", 
                    help="directory of files to evaluate",  
                    required=True)

args=parser.parse_args()

output_dir = args.output_dir

def accuracy(df):
    count = 0
    for _, row in df.iterrows():
        gen = sent_tokenize(row["gen"])[0][len(row["prompt"]):]
        aliases = wikidata_dict[row[f"{row['cf_entity_type']}_id"]]["aliases"]
        if aliases:
            aliases = set(sum([x.split(", ") for x in aliases], []))
        if row[f"{row['cf_entity_type']}_retrieved"] in gen or (aliases and any([a in gen for a in aliases])):
            count += 1
    return count / len(df)

def injection_accuracy_match(df):
    count = 0
    for _, row in df.iterrows():
        gen = sent_tokenize(row["gen_injection"])[0][len(row["prompt"]):]
        aliases = wikidata_dict[row["cf_id"]]["aliases"]
        if aliases:
            aliases = set(sum([x.split(", ") for x in aliases], []))
        if row["cf_entity"] in gen or (aliases and any([a in gen for a in aliases])):
            count += 1
    return count / len(df)

def injection_accuracy_entail(df, batch_size=64):
    entailment_model = "roberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(entailment_model)
    model = AutoModelForSequenceClassification.from_pretrained(entailment_model).to("cuda") 
    texts = []
    count = 0
    for n, row in df.iterrows():
        texts.append(sent_tokenize(row["gen_injection"])[0] + " " + row["cf_full_text"])
        if len(texts) == batch_size or n == len(df)-1:
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                output = model(**inputs)
            for logits in output.logits:
                # proba for contradiction < proba for entailment
                count += 1 if logits[0] < logits[2] else 0
            texts = []
    return count / len(df)

def get_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(sentences, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**inputs)
    embeddings = mean_pooling(output, inputs["attention_mask"])
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu()

def injection_accuracy_embed(df, batch_size=64):
    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model).to("cuda")
    sentences_gen = []
    sentences_gt = []
    count = 0
    for n, row in df.iterrows():
        sentences_gen.append(sent_tokenize(row["gen_injection"])[0][len(row["prompt"]):])
        sentences_gt.append(row["cf_full_text"][len(row["prompt"]):])
        if len(sentences_gen) == batch_size or n == len(df)-1:
            embeddings_gen = get_embeddings(sentences_gen, tokenizer, model)
            embeddings_gt = get_embeddings(sentences_gt, tokenizer, model)
            for embd_gen, embd_gt in zip(embeddings_gen, embeddings_gt):
                cosine_sim = 1 - cosine(embd_gen, embd_gt)
                count += 1 if cosine_sim >= 0.8 else 0
            sentences_gen = []
            sentences_gt = []
    return count / len(df)

llama_chat_template = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST] Answer:"""

system_prompt = """You are given a fact in the form of a (subject, predicate, object) triple and a sentence. Your task is to check if the given fact is present in the sentence.
Answer only with 'Yes' or 'No'."""

user_message_template = """Fact: ({h}, {r}, {t})
Sentence: {sentence}"""

def injection_accuracy_llm(df, batch_size=16):
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    hf_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, padding_side="left")
    tokenizer.pad_token = tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map="auto", 
                                                 load_in_4bit=True, 
                                                 token=hf_token, 
                                                 bnb_4bit_compute_dtype=torch.float16) 
    idx_pos = tokenizer.encode("Yes", add_special_tokens=False)[0]
    idx_neg = tokenizer.encode("No", add_special_tokens=False)[0]
    count = 0
    prompts = []
    for n, row in df.iterrows():
        gen = sent_tokenize(row["gen_injection"])[0]
        if row["cf_entity_type"] == "t":
            user_message = user_message_template.format(
                h=row["h_retrieved"], r=row["r"], t=row["cf_entity"], sentence=gen
            )
        else:
            user_message = user_message_template.format(
                h=row["cf_entity"], r=row["r"], t=row["t_retrieved"], sentence=gen
            )
        prompt = llama_chat_template.format(system_prompt=system_prompt, user_message=user_message)
        prompts.append(prompt)
        if len(prompts) == batch_size or n == len(df) - 1:
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs).logits
            for i, logits in enumerate(outputs):
                # proba for Yes > proba for No
                count += 1 if logits[-1, idx_pos] > logits[-1, idx_neg] else 0
            prompts = []
    return count / len(df)

# not used
def efficacy(df):
    return (df["ppl_post_injection"] < df["ppl_pre_injection"]).mean()

# not used
def efficacy_magnitude(df):
    return (df["ppl_pre_injection"] - df["ppl_post_injection"]).mean()

def fluency_ngram_entropy(df):
    fluency_pre = df.apply(lambda row: compute_n_gram_entropy(sent_tokenize(row["gen"])[0]), axis=1)
    fluency_post = df.apply(lambda row: compute_n_gram_entropy(sent_tokenize(row["gen_injection"])[0]), axis=1)
    return ((fluency_post - fluency_pre) / fluency_pre).mean()

# not used
def fluency_ngram_overlap(df):
    fluency_pre = df.apply(lambda row: seq_rep_n(sent_tokenize(row["gen"])[0]), axis=1)
    fluency_post = df.apply(lambda row: seq_rep_n(sent_tokenize(row["gen_injection"])[0]), axis=1)
    return fluency_post.mean() - fluency_pre.mean()

def predict_acceptability(sentences, tokenizer, model):
    """Predicts grammatical acceptability score of sentence."""
    inputs = tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    probas = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
    return probas[:, 1]
    
def grammatical_correctness(df, full_gen=False, batch_size=64):
    # use complete generation instead of first sentence if full_gen=True
    model_name = "textattack/roberta-base-CoLA"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")
    gen_pre_inj = []
    gen_post_inj = []
    count = 0
    for n, row in df.iterrows():
        if full_gen:
            gen_pre_inj.append(row["gen"])
            gen_post_inj.append(row["gen_injection"])
        else:
            gen_pre_inj.append(sent_tokenize(row["gen"])[0])
            gen_post_inj.append(sent_tokenize(row["gen_injection"])[0])
        if len(gen_pre_inj) == batch_size or n == len(df)-1:
            grammar_pre_inj = predict_acceptability(gen_pre_inj, tokenizer, model)
            grammar_post_inj = predict_acceptability(gen_post_inj, tokenizer, model)
            count += np.sum((grammar_post_inj - grammar_pre_inj) / grammar_pre_inj)
            gen_pre_inj = []
            gen_post_inj = []
    return count / len(df)

def time(df):
    return df["t_injection"].mean()

results = []

files = sorted([file for file in os.listdir(output_dir) if "json" in file])
logger.info(f"Found {len(files)} files in {output_dir}")

for n, fname in enumerate(files):
    logger.info(f"{n+1}/{len(files)} Evaluating {fname}")
    dataset, split, method, model_name = fname.split(".")[0].split("_")

    # load data
    df_gen = pd.read_json(f"{output_dir}/{fname}", lines=True)
    df_gt = pd.read_json(f"../datasets/{dataset}/{dataset}_{split}.json", lines=True)
    df = df_gt.merge(df_gen)
    wikidata_dict = load_wikidata_json("../datasets/wikidata_entity_data.json")
    logger.info(f"Loaded dataframe with {len(df)} rows")
    
    results.append(
        {
            "output_dir": output_dir,
            "fname": fname,
            "dataset": dataset,
            "split": split,
            "method": method,
            "model_name": model_name,
            "model_size": model_size_dict.get(model_name),
            "n_samples": len(df),
            "accuracy": accuracy(df),
            "injection_accuracy_match": injection_accuracy_match(df),
            "injection_accuracy_entail": injection_accuracy_entail(df),
            "injection_accuracy_embed": injection_accuracy_embed(df),
            "injection_accuracy_llm": injection_accuracy_llm(df),
            "fluency": fluency_ngram_overlap(df),
            "grammar": grammatical_correctness(df),
            "time": time(df)  
        }
    )
    logger.info(f"Finished evaluation for file '{fname}'")

logger.info(f"Finished evaluation for {output_dir}")
results_dir = f"{output_dir}_eval.csv"
logger.info(f"Writing results to {results_dir}")
pd.DataFrame(results).to_csv(results_dir, index=False)
