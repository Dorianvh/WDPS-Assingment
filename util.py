import spacy
import subprocess

def write_error(q_id: str, msg:str, path:str, raw_response: str):
     with open(path, 'a') as outfile:
        outfile.write(f'{q_id}\tR"{raw_response}"\n')
        outfile.write(f'{q_id}\tA"{msg}"\n')

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
        
def sparql_generator(triplet):
    pass