import argparse
import contextlib
import logging
import os
import pandas as pd
import subprocess as sp
import sys
import torch
import time
import transformers

sys.path.extend(["../utils", "../methods", "../evaluate", "../methods/mend"])
from constr_beam_search.constr_beam_search_hparams import ConstrBeamSearchHyperParams
from ft.ft_main import apply_ft_to_model
from ft.ft_hparams import FTHyperParams
from in_context.in_context_hparams import InContextHyperParams
from in_context.in_context_main import create_injection_prompt
from mend.mend_hparams import MENDHyperParams
from mend.mend_main import MendRewriteExecutor
from metrics import ppl_generation
from rome.rome_main import apply_rome_to_model
from rome.rome_hparams import ROMEHyperParams
from transformer_utils import load_model, generate
from util import nethook

def get_gpu_memory():
    """ Get used VRAM as shown in nvidia-smi."""
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return sum(memory_free_values)

os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable tokenizer warning

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
parser.add_argument("--compute_ppl", 
                    default=False, 
                    action="store_true",
                    help="output ppl for generation")

args=parser.parse_args()

method = args.method
model_name = args.model_name
dataset = args.dataset
split = args.split
output_dir = args.output_dir
compute_ppl = args.compute_ppl

logger.info("Execute run with parameters:")
logger.info(f"- method:     {method}")
logger.info(f"- model_name: {model_name}")
logger.info(f"- dataset:    {dataset}")
logger.info(f"- split:      {split}")
logger.info(f"- output_dir: {output_dir}")

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

logger.info(f"Load model used VRAM: {get_gpu_memory()}")

logger.info("Load dataset")
df = pd.read_json(f"../datasets/{dataset}/{dataset}_{split}.json", lines=True)

def create_request(row):
    """Create injection request for injection method."""
    request = {
        "prompt": row["prompt"].replace(row["h"] if not row["t_before_h"] else row["t"], "{}", 1),
        "subject": row["h"] if not row["t_before_h"] else row["t"],
        "target_new": {"str": row["cf_entity"]},
        "relation": row["r"]
    }
    return request

results = []
fname = f"{output_dir}/{dataset}_{split}_{method}_{model_name}.json"

logger.info("Start run")
logger.info(f"0/{len(df)} - Writing results to {fname}")

peak_vram = 0

if method == "mend":
    # init mend
    mend_editor = MendRewriteExecutor()

for n, row in df.iterrows():
    
    gen = generate(row["prompt"], tokenizer, model)
    # perplexity for generated part
    if compute_ppl:
        ppl_gen = ppl_generation(model, tokenizer, row["prompt"], gen[len(row["prompt"]):])
    else:
        ppl_gen = None
    # perplexity for cf_full_text without prompt before injection
    # ppl_pre_injection = ppl_generation(model, tokenizer, row["prompt"], row["cf_full_text"][len(row["prompt"]):])

    request = create_request(row)
    
    if method == "in-context":
        t_start = time.time()
        injection_prompt = create_injection_prompt(request, hparams.prompt_template, hparams.split_token)
        gen_injection = generate(injection_prompt, tokenizer, model)
        prompt = injection_prompt.split(hparams.split_token)[-1].strip()
        gen_injection = gen_injection[gen_injection.find(prompt):]
        t_end = time.time()
        # ppl_post_injection = ppl_generation(model, tokenizer, injection_prompt, row["cf_full_text"][len(row["prompt"]):])

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
        gen_injection = generate(row["prompt"], tokenizer, model)
        t_end = time.time()
        # ppl_post_injection = ppl_generation(model, tokenizer, row["prompt"], row["cf_full_text"][len(row["prompt"]):])

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
        gen_injection = generate(row["prompt"], tokenizer, model)
        t_end = time.time()
        # ppl_post_injection = ppl_generation(model, tokenizer, row["prompt"], row["cf_full_text"][len(row["prompt"]):])

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
        gen_injection = generate(row["prompt"], tokenizer, model)
        t_end = time.time()
        # ppl_post_injection = ppl_generation(model, tokenizer, row["prompt"], row["cf_full_text"][len(row["prompt"]):])

    if method == "constr-beam-search":
        t_start = time.time()
        force_words_ids = tokenizer(" " + request["target_new"]["str"], add_special_tokens=False).input_ids
        gen_injection = generate(row["prompt"], tokenizer, model, force_words_ids=[force_words_ids], num_beams=hparams.num_beams)
        t_end = time.time()
        # ppl_post_injection = None

    if method in ["rome", "mend", "ft"]:
        # Restore fresh copy of model
        try:
            with torch.no_grad():
                for k, v in orig_weights.items():
                    nethook.get_parameter(model, k)[...] = v
        except NameError as e:
            logger.warning(f"No model weights to restore: {e}")
    
    results.append(
        {
            "id": row["id"],
            "gen": gen,
            "gen_injection": gen_injection,
            "t_injection": t_end - t_start,
            "ppl_gen": ppl_gen
            # "ppl_pre_injection": ppl_pre_injection,
            # "ppl_post_injection": ppl_post_injection,
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
logger.info(f"Peak used VRAM: {peak_vram}")
logger.info("Finished run")
