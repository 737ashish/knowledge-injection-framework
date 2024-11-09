import json
import numpy as np
import requests
import time

def wikidata_sparql_query(query):
    """
    Post sparql query to wikidata endpoint.
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "research purpose"} # Wikidata policy requires a user-agent header.
    try:
        # .json() should throw error if status code not 200 (OK) e.g., if rate limite
        response = requests.get(url, headers=headers, params={"format": "json", "query": query}).json()
        return response
    except Exception as e:
        return e

def get_wikidata_entity_label(entity_id):
    """
    Returns label for wikidata entity id.
    """
    query = ("SELECT * WHERE {"
             f"wd:{entity_id} rdfs:label ?label."
             "FILTER (langMatches(lang(?label), 'EN'))} LIMIT 1")
    response = wikidata_sparql_query(query)
    try:
        result = response["results"]["bindings"][0]["label"]["value"]
    except:
        result = np.nan
    return result

def get_wikidata_entity_description(entity_id):
    """Returns the description of a Wikidata item."""
    query = ("SELECT distinct ?description WHERE {\n"
            f"wd:{entity_id} schema:description ?description . FILTER (lang(?description)='en')\n"
            "SERVICE wikibase:label { bd:serviceParam wikibase:language 'en' }} LIMIT 100")
    response = wikidata_sparql_query(query)
    result = response["results"]["bindings"]
    if result:
        return result[0]["description"]["value"]
    else:
        return None

def get_wikidata_entity_aliases(entity_id):
    """Returns list of alias from 'as know as' section on wikidata page."""
    query = ("SELECT distinct ?alias WHERE {\n"
            f"wd:{entity_id} skos:altLabel ?alias . FILTER (lang(?alias)='en')\n"
            "SERVICE wikibase:label { bd:serviceParam wikibase:language 'en' }} LIMIT 100")
    response = wikidata_sparql_query(query)
    result = response["results"]["bindings"]
    aliases = [x["alias"]["value"] for x in result]
    if aliases:
        return aliases
    else:
        return None

def get_wikidata_entity_id(entity_name, exact_match=False):
    """
    Function which returns wikidata id for entity name.
    If exact_match=False, first hit on wikidata search is returned. In this case entity_name must not exactly match.
    If exact_match=True, first hit on entity search is returned. In this case there must be an exact match of entity_name.
    Returns None if no id found for entity_name.
    """
    if exact_match:
        params = dict(
            action="wbsearchentities",
            format="json",
            language="en",
            uselang="en",
            search=entity_name)
        response = requests.get("https://www.wikidata.org/w/api.php", params).json() 
        result = response["search"]
    else:
        params = dict(
            action = "query",
            list = "search",
            format = "json",
            uselang="en",
            srsearch = entity_name,
            srprop = "titlesnippet|snippet", 
            srlimit = 1)
        response = requests.get("https://www.wikidata.org/w/api.php", params).json()
        result = response["query"]["search"]
    if result:
        return result[0]["title"]
    else:
        return None

def get_wikidata_property_id(property_name):
    """
    Function returns wikidata property id for property name.
    Property name should be exact match because REBEL model used for IE was trained on Wikidata properties.
    """
    params = dict(
        action = "wbsearchentities", 
        format = "json", 
        language = "en", 
        uselang = "en",
        type = "property", 
        search = property_name
    )
    response = requests.get("https://www.wikidata.org/w/api.php?", params).json()
    result = response["search"]
    if result:
        return result[0]["id"]
    else:
        return None

def query_triple_tail(entity_id, property_id):
    """
    Function returns list of entities with id that match (H, P, ?) triple SPARQL query.
    If no such triple exists, None is returned.
    """
    query = ("SELECT ?item ?itemLabel\n"
        "WHERE {\n"
        f"wd:{entity_id} wdt:{property_id} ?item.\n"
        "SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. }} LIMIT 1")
    response = wikidata_sparql_query(query)
    result = response["results"]["bindings"]
    if result:
        tail_ids = [x["item"]["value"].split("/")[-1] for x in result]
        tail_names = [x["itemLabel"]["value"] for x in result]

        # format dates if entity is date
        try:
            tail_names = [parse_date(x) for x in tail_names]
        except:
            pass

        return [(x, y) for x, y in zip(tail_names, tail_ids)]
    else:
        return None

def query_triple_head(entity_id, property_id):
    """
    Function returns list of entities with id that match (?, P, T) triple SPARQL query.
    If no such triple exists, None is returned.
    """
    query = ("SELECT ?item ?itemLabel\n"
        "WHERE {\n"
        f"?item wdt:{property_id} wd:{entity_id}.\n"
        "SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. }} LIMIT 1")
    response = wikidata_sparql_query(query)
    result = response["results"]["bindings"]
    if result:
        head_ids = [x["item"]["value"].split("/")[-1] for x in result]
        head_names = [x["itemLabel"]["value"] for x in result]
        return [(x, y) for x, y in zip(head_names, head_ids)]
    else:
        return None

def load_wikidata_json(path):
    "Loads dict from json of wikidata entity data saved with pd.to_json(orient='records', lines=True)."
    with open(path, "r") as file:
        data = [json.loads(line) for line in file]
    wikidata_dict = {item["id"]: {
        "label": item["label"],
        "description": item["description"],
        "aliases": item["aliases"]
    } for item in data}
    return wikidata_dict

def parse_date(input_date):
    """Parses date from '1961-08-04T00:00:00Z' to 'August 4, 1961' format."""
    date_obj = datetime.strptime(input_date, "%Y-%m-%dT%H:%M:%SZ")
    formatted_date = date_obj.strftime("%B %d, %Y")
    return formatted_date
