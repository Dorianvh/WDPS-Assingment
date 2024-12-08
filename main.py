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

def ensure_model_installed(model_name="en_core_web_md"):
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
    
    nlp = spacy.load("en_core_web_md")
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
         
def classify_question(doc):
    """
    Classifies a question as requiring a 'YES/NO' or 'entity' answer.

    Args:
        doc (spacy.tokens.Doc): A SpaCy Doc object representing the parsed question.

    Returns:
        str: 'YES/NO' or 'entity' based on the analysis.
    """
    # Retrieve the root of the sentence
    root = [token for token in doc if token.dep_ == "ROOT"][0]
    
    # Check for auxiliary verbs indicating YES/NO questions
    if root.pos_ == "AUX" and root.lemma_ in {"is", "does", "are", "was", "were"}:
        return "YES/NO"

    # Check for pronouns like "Who" or "What" at the root or as subject
    if any(token.text.lower() in {"who", "what", "where", "when"} for token in doc if token.dep_ in {"attr", "nsubj"}):
        return "ENTITY"

    # Default to YES/NO if no specific rules match
    return "YES/NO"       
      
def extract_yes_no(answer):
    if answer.strip().lower().startswith('yes'):
        return 'yes'
    if answer.strip().lower().startswith('no'):
        return 'no'
    else:
        return 'answer makes no sense. (couldnt find an affirmative or negative statement)'

def extract_answer_entity(answer):
    # Process the sentence with spaCy
    nlp=spacy.load('en_core_web_md')
    doc = nlp(answer)

    n_words = sum(1 for token in doc if token.is_alpha)
    if n_words == 1:
        return answer
    

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

    if not nominal_subject and not passive_nominal_subject:
        # Find all named entities in the sentence
        named_entities = [ent for ent in doc.ents]
        if len(named_entities) == 0:
            return'answer makes no sense (there are no named entities and no nominal subject)'
        elif len(named_entities) == 1:
            return named_entities[0]
        else:
            return 'answer makes no sense (there are multiple named entities but no subject)'

    elif passive_nominal_subject:
        named_entities = [(ent, abs(passive_nominal_subject.i - ent.start)) for ent in doc.ents]
        if named_entities:
            # Return the named entity closest to the nominal subject
            closest_entity = min(named_entities, key=lambda x: x[1])[0]
            return closest_entity
        else:
            return 'answer makes no sense (there are no named entities but there is a passive nominal subject)'
        
    else:
        named_entities = [(ent, abs(nominal_subject.i - ent.start)) for ent in doc.ents]
        if named_entities:
            # Return the named entity closest to the nominal subject
            closest_entity = min(named_entities, key=lambda x: x[1])[0]
            return closest_entity
        else:
            return 'answer makes no sense (there are no named entities but there is an nominal subject)'  
            
def main():
    
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        epilog='Text at the bottom of help')
    parser.add_argument('-infile','-if')
    parser.add_argument('-outfile','-of')
    args = parser.parse_args()
    
    #check if the required model is installed and if not, download it
    ensure_model_installed()
    questions = read_input(args.infile)
    
    #create or wipe the output file
    with open(args.outfile, 'w') as f:
        pass
    
    #process the questions
    for q_id, q_text in questions.items():

        nlp=spacy.load('en_core_web_md')
        doc = nlp(q_text)
        expected_answer_type = classify_question(doc)

        #fetch llm output
        raw_answer = ask_question(q_text)

        raw_answer = raw_answer.lstrip(': ')

        if expected_answer_type == 'YES/NO':
            extracted_answer= extract_yes_no(raw_answer)
        if expected_answer_type == 'ENTITY':
            extracted_answer= extract_answer_entity(raw_answer)
        
        #extract entities and disambiguate
        entities = recognize_entities(raw_answer)
        links = link_entities(entities=entities)
        
        #append results to outfile
        append_outfile(path=args.outfile,
                       q_id=q_id,
                       raw_response=raw_answer,
                       answer=extracted_answer,
                       entities=links)
        
if __name__ == '__main__':
    
    main()