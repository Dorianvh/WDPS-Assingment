import spacy
import argparse
import subprocess
from llama_cpp import Llama

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
    # Load the pre-trained spaCy model
    nlp = spacy.load("en_core_web_sm")
    # Process the text
    doc = nlp(text)
    # Extract entities
    entities = [ent.text for ent in doc.ents]
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
        for entity, url in entities:
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
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('-filename','-f')
    
    ensure_model_installed()
    
    args = parser.parse_args()
    questions = read_input(args.filename)
    for q_id, q_text in questions.items():
        
        answer = ask_question(question=q_text)
        entities = recognize_entities(answer)
        print(entities)
        
        append_outfile(path='code/output.txt',
                       q_id=q_id,
                       raw_response=answer,
                       entities=[(entity, 'www.blabla.com') for entity in entities])
        
if __name__ == '__main__':
    main()
