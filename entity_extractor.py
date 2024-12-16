import requests
import spacy


def extract_answer_entity(answer, linked_entities):
    # Process the sentence with spaCy
    nlp=spacy.load('en_core_web_md')
    doc = nlp(answer)
    
    #check if the answer has several sentences and prioritize the first senctence 
    if len(list(doc.sents)) >=2:
        output = extract_answer_entity(next(doc.sents).text, linked_entities)
        if isinstance(output, tuple):
            return output
        
    output = None

    
    # Find the nominal subject (nsubj)
    nominal_subject = None
    passive_nominal_subject = None

    #try to find a subject first
    for token in doc:
        if token.dep_ == "nsubj":
            nominal_subject = token
            break

    #try to find a passive subject second
    if not nominal_subject:
        for token in doc:
            if token.dep_ == "nsubjpass":
                nominal_subject = token
                break
            
    n_words = sum(1 for token in doc if token.is_alpha)
    if n_words == 1:
        output = answer

    elif not nominal_subject and not passive_nominal_subject:
        # Find all named entities in the sentence
        named_entities = [ent for ent in get_filtered_entities(doc)]
        if len(named_entities) == 0:
            return'answer makes no sense (there are no named entities and no nominal subject)', ""
        else:
            output = named_entities[0].text

    elif passive_nominal_subject:
        named_entities = [(ent, abs(passive_nominal_subject.i - ent.start)) for ent in get_filtered_entities(doc)]
        if named_entities:
            # Return the named entity closest to the nominal subject
            closest_entity = min(named_entities, key=lambda x: x[1])[0]
            output = closest_entity.text
        else:
            return 'answer makes no sense (there are no named entities but there is a passive nominal subject)', ""
        
    else:
        named_entities = [(ent, abs(nominal_subject.i - ent.start)) for ent in get_filtered_entities(doc)]
        if named_entities:
            # Return the named entity closest to the nominal subject
            closest_entity = min(named_entities, key=lambda x: x[1])[0]
            output = closest_entity.text
        else:
            return 'answer makes no sense (there are no named entities but there is an nominal subject)' , "" 

    #first try to see if we can find the entity based on the cleaned label and return that
    for _, entity_label, entity_url in linked_entities:
        if entity_label is None:
            continue
        if output in entity_label or entity_label in output:
            return entity_label, entity_url
        
    #next try to see if we can atleast find th entity based on the raw label (maybe it mismatched like 'apple' and 'APPL')
    for entity_raw, _, entity_url in linked_entities:
        if output in entity_raw or entity_raw in output:
            return entity_label, entity_url
    #finally just return the output if nothing else matches
    return output, output


def get_filtered_entities(doc):
    
    filter = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    return [ent for ent in doc.ents if ent.label_ not in filter]

def recognize_entities(text):
    """
    Perform Named Entity Recognition (NER) on the given text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of recognized entities with their labels.
    """
    
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    
    return [ent.text for ent in get_filtered_entities(doc)]

def link_entities(entities):
    linked_entities = []
    for entity in entities:
        candidates = generate_candidates_api(entity)
        if candidates:
            best_candidate = None
            max_sitelinks = -1
            for candidate in candidates:
                entity_info = get_entity_info(candidate)
                if entity_info['sitelinks'] > max_sitelinks:
                    best_candidate = entity_info
                    max_sitelinks = entity_info['sitelinks']
            linked_entities.append((entity, best_candidate['label'], best_candidate['url']))
        else:
            linked_entities.append((entity, None, None))
    return linked_entities

def generate_candidates_api(mention, language="en", limit=10):
    url = "https://www.wikidata.org/w/api.php"

    params = {
        "action": "wbsearchentities",
        "search": mention,
        "language": language,
        "format": "json",
        "limit": limit,
    }

    response = requests.get(url, params=params)
    data = response.json()

    candidates = []

    for entity in data.get("search", []):
        candidates.append(entity["id"])
    return candidates

def get_entity_info(id, languages='en'):
    url = "https://www.wikidata.org/w/api.php"

    params = {
        "action": "wbgetentities",
        "ids": id,
        "languages": languages,
        "format": "json",
    }

    response = requests.get(url, params=params)
    data = response.json()

    label = data['entities'][id]['labels'].get('en', {}).get('value', 'No label available')
    description = data['entities'][id]['descriptions'].get('en', {}).get('value', 'No description available')
    claims = len(data['entities'][id]['claims'].keys())
    sitelinks = len(data['entities'][id]['sitelinks'].keys())

    url = ''

    if data['entities'][id]['sitelinks'].get('enwiki'):
        base_url = 'https://en.wikipedia.org/wiki/'
        url = base_url + data['entities'][id]['sitelinks']['enwiki']['title'].replace(' ', '_')

    return {'label': label, 'description': description, 'claims': claims, 'sitelinks': sitelinks, 'url': url}