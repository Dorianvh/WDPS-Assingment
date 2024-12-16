import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import pipeline
from entity_extractor import extract_answer_entity

# Function to parse the generated text and extract the triplets
def extract_triplets(input_text):
    
    triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')
    # We need to use the tokenizer manually since we need special tokens.
    extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(input_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])

    text = extracted_text[0]
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

def get_wikidata_id(label: str) -> str:
    """
    Retrieves the Wikidata ID for an entity or property based on its label.
    
    Args:
        label (str): The label of the entity or property (e.g., "Human", "Instance of").
    
    Returns:
        str: The Wikidata ID (e.g., "Q5" for Human, "P31" for Instance of), or None if not found.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": label
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("search", [])
        
        if results:
            return results[0]["id"]
        else:
            print(f"No results found for label: {label}")
            return None
    except Exception as e:
        print(f"Error retrieving Wikidata ID: {e}")
        return None

def get_property_id(label: str) -> str:
    """
    Retrieves the Wikidata ID for an entity or property based on its label.
    
    Args:
        label (str): The label of the entity or property (e.g., "Human", "Instance of").
    
    Returns:
        str: The Wikidata ID (e.g., "Q5" for Human, "P31" for Instance of), or None if not found.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "type": "property",
        "language": "en",
        "format": "json",
        "search": label
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("search", [])
        
        if results:
            return results[0]["id"]
        else:
            print(f"No results found for label: {label}")
            return None
    except Exception as e:
        print(f"Error retrieving Wikidata ID: {e}")
        return None

def is_property_entailed(entity_label1: str, entity_label2: str, property_label: str) -> bool:
    """
    Checks if a given Wikidata property entails a relationship between two entities.
    
    Args:
        entity_label1 (str): Label of the first entity (e.g., "Douglas Adams").
        entity_label2 (str): Label of the second entity (e.g., "Human").
        property_label (str): Label of the property (e.g., "Instance of").
    
    Returns:
        bool: True if the property entails a relationship, False otherwise.
    """
    print(entity_label1)
    print(entity_label2)
    print(property_label)
    # Get the IDs for the entities and property
    entity1 = get_wikidata_id(entity_label1)
    entity2 = get_wikidata_id(entity_label2)
    property_id = get_property_id(property_label)
    
    print(entity1)
    print(entity2)
    print(property_id)

    if not entity1 or not entity2 or not property_id:
        print("Could not resolve one or more labels to Wikidata IDs.")
        return False

    # Define the SPARQL query
    sparql_query = f"""
    ASK {{
      wd:{entity1} wdt:{property_id} wd:{entity2} .
    }}
    """

    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    try:
        # Execute the query
        response = sparql.query().convert()
        return response.get("boolean", False)
    except Exception as e:
        print(f"Error querying SPARQL: {e}")
        return False
    
def extract_candidate_fact(triplets, entities, text):
    
    #if there is only one triple, return it raw
    if len(triplets) == 1 and not entities:
        return triplets[0]
    
    #if there is only one triple but we have entities, return that triple with the clean label of the linked entity
    elif len(triplets) == 1 and entities:
        triple = triplets[0]
        for raw_ent, ent_label, _ in entities:
            if triple['head'] in raw_ent or raw_ent in triple['head']:
                return {'head': ent_label, 'type': triple['type'], 'tail':triple['tail']}
        
        return triplets[0]
    
    #if there are multiple triples, find the one with the most suitable type or 'property' and return with the label
    elif len(triplets) >= 2 and entities:
        for triple in triplets:
            if triple['type'] in text:
                for raw_ent, ent_label, _ in entities:
                    if triple['head'] in raw_ent or raw_ent in triple['head']:
                        return {'head': ent_label, 'type': triple['type'], 'tail':triple['tail']}
    
    #if there are no multiple triples but no entities, return the most suitable triple raw
    elif len(triplets) >= 2 and not entities:
        for triple in triplets:
            if triple['type'] in text:
                return triple

    #if there are no entities and we couldn't find an appropriate triple based on type, find the first triple where the head is the subject of the answer
    else:
        subject = extract_answer_entity(text)
        for triple in triplets:
            if triple['head'] in subject or subject in triple['head']:
                return triple
    
    #hail mary time
    return triplets[0]
    
    
#extracted_triplets = extract_triplets("Is Managua the Capital of Nicaragua?")
#for t in extracted_triplets:
#    entity_label1 = t['head']
#    entity_label2 = t['tail']
#    property_label = t['type']
##    # Check if the property entails the relationship
#    result = is_property_entailed(entity_label1, entity_label2, property_label)
#    print(f"Does '{property_label}' entail a relationship between '{entity_label1}' and '{entity_label2}'? {result}")
    


