import re
import random
from collections import Counter
from datetime import datetime
from wikidata import get_wikidata_entity_id, get_wikidata_property_id, query_triple_tail, query_triple_head

def sample_item(l, random_seed=42):
    """Sample item from list."""
    random.seed(random_seed)
    return random.choice(l)

def get_unique_dict_items(l, counts=False):
    """Returns all unique dicts in a list of dicts. If counts=True new key with counts is appended to each dict."""
    unique_tuples = Counter([tuple(x.items()) for x in l]).most_common()
    return [dict(x[0]) if not counts else dict(x[0]) | dict(counts=x[1]) for x in unique_tuples]

def augment_wikidata(triples, gen, prompt):
    """Entity linking and entity query for extracted entity in completion."""
    triples = [x.copy() for x in triples]
    for triple in triples:
        h, r, t = triple["h"], triple["r"], triple["t"]
        # head entity in prompt, tail entity in completion -> retrieve tail entity
        if h in prompt and t in gen[len(prompt):]:
            if h_id := get_wikidata_entity_id(h, exact_match=True):
                if r_id := get_wikidata_property_id(r):
                    if t_retrieved := query_triple_tail(h_id, r_id):
                        triple["retrieved"] = t_retrieved
                        triple["entity_type"] = "t"
                        break # stop retrieval since we use majority triple anyways and order is by descending counts
        # tail entity in prompt, head entity in completion -> retrieve head entity
        elif t in prompt and h in gen[len(prompt):]:
            if t_id := get_wikidata_entity_id(t, exact_match=True):
                if r_id := get_wikidata_property_id(r):
                    if h_retrieved := query_triple_head(t_id, r_id):
                        triple["retrieved"] = h_retrieved
                        triple["entity_type"] = "h"
                        break
    return triples

def triple_candidate_selection(triples):
    """Returns majority triple with available wikidata tail entity from a list of knowledge augement triples."""
    # drop all triples for which no tail entity could be retrieved
    triple_candidates = [x for x in triples if "retrieved" in x.keys()]
    # select majority triple as final triple
    if triple_candidates:
        triple = max(triple_candidates, key=lambda x: (x["counts"], -triple_candidates.index(x)))
        return triple
    else:
        return None
    