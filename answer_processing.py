from transformers import pipeline
from llama_cpp import Llama

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
    
    #First try to extract simple answers without having to use the complex classifier to save time and resources.
    if answer.strip().lower().startswith('yes'):
        return 'yes'
    if answer.strip().lower().startswith('no'):
        return 'no'
    
    #if not, use the zer-shot classifier
    
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["yes", "no"]
    result = classifier(answer, labels)
    
    # Extract scores
    scores = result["scores"]
    labels = result["labels"]
    best_label = labels[0]
    best_score = scores[0]
    
    # Add a threshold for confidence
    if best_score > 0.6:  # Confidence threshold
        return best_label
    
    else:
        return 'answer makes no sense. (couldnt find an affirmative or negative statement)'