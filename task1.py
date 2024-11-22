import nltk
from nltk.corpus import wordnet as wn
import spacy
from pprint import pprint
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util

nltk.download('wordnet')

# Load NER model
nlp = spacy.load("en_core_web_sm")

# Load in Questions/Answers
QaA = []

model = SentenceTransformer('all-MiniLM-L6-v2')


with open('QuestionsAndAnswers.txt', mode='r',encoding="utf8") as file:
    for line in file:
        q, a = line.strip().split(':/:')

        QaA.append({'question': q, 'answer': a})

# List of named entities
# PERSON:      People, including fictional.
# NORP:        Nationalities or religious or political groups.
# FAC:         Buildings, airports, highways, bridges, etc.
# ORG:         Companies, agencies, institutions, etc.
# GPE:         Countries, cities, states.
# LOC:         Non-GPE locations, mountain ranges, bodies of water.
# PRODUCT:     Objects, vehicles, foods, etc. (Not services.)
# EVENT:       Named hurricanes, battles, wars, sports events, etc.
# WORK_OF_ART: Titles of books, songs, etc.
# LAW:         Named documents made into laws.
# LANGUAGE:    Any named language.
# DATE:        Absolute or relative dates or periods.
# TIME:        Times smaller than a day.
# PERCENT:     Percentage, including ”%“.
# MONEY:       Monetary values, including unit.
# QUANTITY:    Measurements, as of weight or distance.
# ORDINAL:     “first”, “second”, etc.
# CARDINAL:    Numerals that do not fall under another type.

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


def link_entities_with_embeddings(entities, doc):
    linked_entities = []
    for entity in entities:
        candidates = generate_candidates_api(entity)
        if candidates:
            best_candidate = None
            max_similarity = -1

            # Find the sentence containing the entity
            entity_sentence = next(sent for sent in doc.sents if entity in sent.text)
            entity_embedding = model.encode(entity_sentence.text, convert_to_tensor=True)

            for candidate in candidates:
                entity_info = get_entity_info(candidate)
                candidate_embedding = model.encode(entity_info['description'], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(entity_embedding, candidate_embedding).item()
                if similarity > max_similarity:
                    best_candidate = entity_info
                    max_similarity = similarity
            linked_entities.append((entity, best_candidate['label'], best_candidate['url']))
        else:
            linked_entities.append((entity, None, None))
    return linked_entities


NER = []

for qa in QaA:
    q_doc, a_doc = nlp(qa['question']), nlp(qa['answer'])

    dont_include_labels = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

    q_entities = [entity.text for entity in q_doc.ents if entity.label_ not in dont_include_labels]
    a_entities = [entity.text for entity in a_doc.ents if entity.label_ not in dont_include_labels]

    #linked_q_entities = link_entities_with_embeddings(q_entities, q_doc)
    #linked_a_entities = link_entities_with_embeddings(a_entities, a_doc)

    linked_q_entities = link_entities(q_entities)
    linked_a_entities = link_entities(a_entities)

    print(f"Question: {qa['question']}")
    print(f"Linked Entities in Question: {linked_q_entities}")
    print(f"Answer: {qa['answer']}")
    print(f"Linked Entities in Answer: {linked_a_entities}")
    print()