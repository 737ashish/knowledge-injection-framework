import argparse
import contextlib
import json
import logging
import os
import pandas as pd
import subprocess as sp
import sys
import torch
import time
import transformers

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.extend(["../utils", "../methods", "../methods/mend"])
from constr_beam_search.constr_beam_search_hparams import ConstrBeamSearchHyperParams
from ft.ft_main import apply_ft_to_model
from ft.ft_hparams import FTHyperParams
from in_context.in_context_hparams import InContextHyperParams
from in_context.in_context_main import create_injection_prompt
from information_extraction import extract_triples
from mend.mend_hparams import MENDHyperParams
from mend.mend_main import MendRewriteExecutor
from pipeline_utils import get_unique_dict_items, augment_wikidata, triple_candidate_selection
from rome.rome_main import apply_rome_to_model
from rome.rome_hparams import ROMEHyperParams
from transformer_utils import load_model, generate
from util import nethook
from wikidata import get_wikidata_entity_aliases

os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable tokenizer warning

def get_gpu_memory():
    """ Get used VRAM as shown in nvidia-smi."""
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return sum(memory_free_values)

# set up logging
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

parser = argparse.ArgumentParser()
parser.add_argument("--method", 
                    help="knowledge injection method", 
                    choices=["in-context", "rome", "mend", "ft", "constr-beam-search"], 
                    required=True)
parser.add_argument("--model_name", 
                    help="language model", 
                    choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "gpt-j-6b", "llama-2-7b", "llama-2-13b", "llama-2-70b"], 
                    required=True)
parser.add_argument("--dataset", 
                    help="evaluation dataset", 
                    choices=["fewrel", "counterfact"], 
                    required=True)
parser.add_argument("--split", 
                    help="split of dataset", 
                    choices=["dev", "val"], 
                    required=True)
parser.add_argument("--output_dir", 
                    help="directory to store results", 
                    required=True)
parser.add_argument("--persist_wikidata", 
                    default=False, 
                    action="store_true",
                    help="save/load queried wikidata triples locally")
parser.add_argument("--injection_model",
                    help="use injection model for in-context knowledge injection")
parser.add_argument("--retrieve_gt", 
                    default=False, 
                    action="store_true",
                    help="rietrieve ground truth instead of querying wikidata")

args=parser.parse_args()

method = args.method
model_name = args.model_name
dataset = args.dataset
split = args.split
output_dir = args.output_dir
persist_wikidata = args.persist_wikidata
injection_model = args.injection_model
retrieve_gt = args.retrieve_gt

logger.info("Execute pipeline with parameters:")
logger.info(f"- method:     {method}")
logger.info(f"- model_name: {model_name}")
logger.info(f"- dataset:    {dataset}")
logger.info(f"- split:      {split}")
logger.info(f"- output_dir: {output_dir}")
logger.info(f"- persist_wikidata: {persist_wikidata}")
logger.info(f"- injection_model: {injection_model}")
logger.info(f"- retrieve_gt: {retrieve_gt}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger.info("Load injection method hyperparameters")
if method == "in-context":
    hparams = InContextHyperParams.from_json(f"../methods/hparams/{method}/{model_name}.json")
elif method == "rome":
    hparams = ROMEHyperParams.from_json(f"../methods/hparams/{method}/{model_name}.json")
elif method == "mend":
    hparams = MENDHyperParams.from_json(f"../methods/hparams/{method}/{model_name}.json")
elif method == "ft":
    hparams = FTHyperParams.from_json(f"../methods/hparams/{method}/{model_name}.json")
elif method == "constr-beam-search":
    hparams = ConstrBeamSearchHyperParams.from_json(f"../methods/hparams/{method}/{model_name}.json")
else:
    raise NotImplementedError

logger.info(f"Init used VRAM: {get_gpu_memory()}")

logger.info("Load tokenizer and model")
tokenizer, model = load_model(hparams.hf_model)

# use different model for injection when using in-context knowledge injection
if injection_model:
    inj_hparams = InContextHyperParams.from_json(f"../methods/hparams/{method}/{injection_model}.json")
    logger.info("Load tokenizer and injection model for in-context knowledge injection")
    inj_tokenizer, inj_model = load_model(inj_hparams.hf_model)

logger.info(f"Load model used VRAM: {get_gpu_memory()}")

logger.info("Load dataset")
df = pd.read_json(f"../datasets/{dataset}/{dataset}_{split}.json", lines=True)

# triple extraction model
logger.info("Load information extraction model")
ie_model_name = "Babelscape/rebel-large"
ie_tokenizer = AutoTokenizer.from_pretrained(ie_model_name)
ie_model = AutoModelForSeq2SeqLM.from_pretrained(ie_model_name).to("cuda")

def create_request(prompt, triple):
    """Create injection request for injection method."""
    request = {
        "prompt": prompt.replace(triple["h"] if triple["entity_type"] == "t" else triple["t"], "{}", 1),
        "subject": triple["h"] if triple["entity_type"] == "t" else triple["t"],
        "target_new": {"str": triple["retrieved"][0][0]}, # use first retrieved entity, query LIMIT currently set to 1
        "relation": triple["r"]
    }
    return request

def get_aliases(triple):
    try: 
        aliases = get_wikidata_entity_aliases(triple["retrieved"][0][1])
    except:
        logger.info(f"Wikidata alias query failed for {triple['retrieved'][0]}.")
        aliases = []
    if aliases:
        aliases = list(set(sum([x.split(", ") for x in aliases], [])))
    else:
        aliases = []
    return aliases

results = []
fname = f"{output_dir}/pipeline_{dataset}_{split}_{method}_{model_name}.json"

fname_triples = f"data/wikidata/{dataset}_{split}_{method}_{model_name}_triples.json"
if persist_wikidata and os.path.exists(fname_triples):
    logger.info(f"Load wikidata triples file from {fname_triples}")
    with open(fname_triples, "r") as json_file:
        wikidata_triples = json.load(json_file)
    assert len(wikidata_triples) == len(df)
elif persist_wikidata and not os.path.exists(fname_triples):
    logger.info(f"Persist wikidata triples to {fname_triples}")
    wikidata_triples = None
    wikidata_triples_dict = {}
else:
    wikidata_triples = None

logger.info("Start run")
logger.info(f"0/{len(df)} - Writing results to {fname}")

peak_vram = 0

if method == "mend":
    # init mend
    mend_editor = MendRewriteExecutor()

for n, row in df.iterrows():

    t_start_pipeline = time.time()
    
    t_start = time.time()
    gen = sent_tokenize(generate(row["prompt"], tokenizer, model))[0]
    t_gen = time.time() - t_start
    
    # triple extraction
    t_start = time.time()
    triples_extracted = extract_triples(gen, ie_tokenizer, ie_model)
    t_extraction = time.time() - t_start
    triples_unique = get_unique_dict_items(triples_extracted, counts=True)
    
    # entity linking and information retrieval
    if not wikidata_triples:
        t_start = time.time()
        triples_augmented = augment_wikidata(triples_unique, gen, row["prompt"])
        if persist_wikidata:
            wikidata_triples_dict[row["id"]] = {"triples_augmented": triples_augmented}
    else:
        t_start = time.time()
        triples_augmented = wikidata_triples[f"{row['id']}"]["triples_augmented"]
    triple = triple_candidate_selection(triples_augmented)
    
    # fact checking
    if triple and triple["entity_type"] == "t":
        aliases = get_aliases(triple)
        correction = triple["retrieved"][0][0] != triple["t"] and triple["t"] not in aliases
    elif triple and triple["entity_type"] == "h":
        aliases = get_aliases(triple)
        correction = triple["retrieved"][0][0] != triple["h"] and triple["h"] not in aliases
    else:
        correction = False
    t_retrieval = time.time() - t_start

    if correction:

        # retrieve triple to inject from ground truth
        if retrieve_gt:
            triple = {
                "h": row["h"] if row["cf_entity_type"] == "t" else row["h_retrieved"],
                "r": row["r"],
                "t": row["t"] if row["cf_entity_type"] == "h" else row["t_retrieved"],
                "retrieved": [(row[f"{row['cf_entity_type']}_retrieved"], row[f"{row['cf_entity_type']}_id"])],
                "entity_type": row["cf_entity_type"]
            }
        
        request = create_request(row["prompt"], triple)

        t_start = time.time()
        
        if method == "in-context":
            if injection_model:
                injection_prompt = create_injection_prompt(request, inj_hparams.prompt_template, inj_hparams.split_token)
                gen_pipeline = generate(injection_prompt, inj_tokenizer, inj_model, top_p=1).split(inj_hparams.split_token, 1)[-1].strip()
            else:
                injection_prompt = create_injection_prompt(request, hparams.prompt_template, hparams.split_token)
                gen_pipeline = generate(injection_prompt, tokenizer, model).split(hparams.split_token, 1)[-1].strip()
            t_injection = time.time() - t_start

        if method == "rome":
            t_start = time.time()
            with contextlib.redirect_stdout(None):
                model, orig_weights = apply_rome_to_model(
                        model=model,
                        tok=tokenizer,
                        requests=[request],
                        hparams=hparams,
                        copy=False,
                        return_orig_weights=True
                )
            gen_pipeline = generate(row["prompt"], tokenizer, model)
        
        if method == "mend":
            t_start = time.time()
            model, orig_weights = mend_editor.apply_to_model(
                model=model,
                tok=tokenizer,
                requests=[request],
                hparams=hparams,
                copy=False,
                return_orig_weights=True
            )
            gen_pipeline = generate(row["prompt"], tokenizer, model)
           
        if method == "ft":
            t_start = time.time()
            with contextlib.redirect_stdout(None):
                model, orig_weights = apply_ft_to_model(
                    model=model,
                    tok=tokenizer,
                    requests=[request],
                    hparams=hparams,
                    copy=False,
                    return_orig_weights=True
                )
            gen_pipeline = generate(row["prompt"], tokenizer, model)
        
        if method == "constr-beam-search":
            force_words_ids = tokenizer(" " + request["target_new"]["str"], add_special_tokens=False).input_ids
            gen_pipeline = generate(row["prompt"], tokenizer, model, force_words_ids=[force_words_ids], num_beams=hparams.num_beams)
        
        if method in ["rome", "mend", "ft"]:
            # Restore fresh copy of model
            try:
                with torch.no_grad():
                    for k, v in orig_weights.items():
                        nethook.get_parameter(model, k)[...] = v
            except NameError as e:
                logger.warning(f"No model weights to restore: {e}")

        t_injection = time.time() - t_start
            
    else:
        gen_pipeline = gen
        t_injection = 0
    
    t_pipeline = time.time() - t_start_pipeline

    results.append(
        {
            "id": row["id"],
            "gen": gen,
            "triple_retrieved": triple,
            "correction": correction,
            "gen_pipeline": gen_pipeline,
            "t_gen": t_gen,
            "t_extraction": t_extraction,
            "t_retrieval": t_retrieval,
            "t_injection": t_injection,
            "t_pipeline": t_pipeline
        }
    )
        
    if (n+1) % 100 == 0:
        logger.info(f"{n+1}/{len(df)} - Writing results to {fname}")
        pd.DataFrame(results).to_json(fname, orient="records", lines=True, mode="a")
        results = []

        if peak_vram < get_gpu_memory():
            peak_vram = get_gpu_memory()

logger.info(f"{n+1}/{len(df)} - Writing results to {fname}")
pd.DataFrame(results).to_json(fname, orient="records", lines=True, mode="a")

if persist_wikidata and not os.path.exists(fname_triples):
    logger.info(f"Writing wikidata triples to {fname_triples}")
    with open(fname_triples, "w") as json_file:
        json.dump(wikidata_triples_dict, json_file)

logger.info(f"Peak used VRAM: {peak_vram}")
logger.info("Finished run")
