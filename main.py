import spacy
import argparse
import subprocess
import requests
from llama_cpp import Llama

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

def ensure_model_installed(model_name="en_core_web_sm"):
    """
    Ensure the specified spaCy model is installed. Install it if not available.

    Args:
        model_name (str): Name of the spaCy model to check or install.
    """
    try:
        # Try to load the model to check if it's installed
        spacy.load(model_name)
        print(f"Model '{model_name}' is already installed.")
    except OSError:
        print(f"Model '{model_name}' not found. Installing...")
        # Install the model using subprocess
        subprocess.check_call(["python", "-m", "spacy", "download", model_name])
        print(f"Model '{model_name}' installed successfully.")
          
def recognize_entities(text):
    """
    Perform Named Entity Recognition (NER) on the given text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of recognized entities with their labels.
    """
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    filter = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    entities = [ent.text for ent in doc.ents if ent.label_ not in filter]
    
    return entities

def read_input(path):
    
    output = dict()
    
    with open(path, 'r') as infile:
        content = infile.readlines()
        
    for line in content:
        
        if not line.startswith('question-'):
            continue
        
        q_id, q_text = line.split('\t')
        output[q_id] = q_text.rstrip('\n')

    return output

def append_outfile(path:str,
                   q_id:str,
                   raw_response:str,
                   answer:str = 'N/A',
                   correctness:str = 'N/A',
                   entities:list = []
                   ):
    
    with open(path, 'a') as outfile:
        outfile.write(f'{q_id}\tR"{raw_response}"\n')
        outfile.write(f'{q_id}\tA"{answer}"\n')
        outfile.write(f'{q_id}\tC"{correctness}"\n')
        for entity, _, url in entities:
            outfile.write(f'{q_id}\tE"{entity}"\t"{url}"\n')
    
def ask_question(question, model_path = "models/llama-2-7b.Q4_K_M.gguf"):
    
    llm = Llama(model_path=model_path, verbose=False)
    print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))
    output = llm(
        question, # Prompt
        max_tokens=32, # Generate up to 32 tokens
        stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
        echo=False # Echo the prompt back in the output
    )
    return output['choices'][0]['text']        
            
def main():
    
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        epilog='Text at the bottom of help')

    parser.add_argument('-infile','-if')
    parser.add_argument('-outfile','-of')
    args = parser.parse_args()
    
    ensure_model_installed()
    questions = read_input(args.infile)
    
    with open(args.outfile, 'w') as f:
        pass
    
    for q_id, q_text in questions.items():
        
        answer = ask_question(question=q_text)
        entities = recognize_entities(answer)
        links = link_entities(entities=entities)
        append_outfile(path=args.outfile,
                       q_id=q_id,
                       raw_response=answer,
                       entities=links)
        
if __name__ == '__main__':
    main()