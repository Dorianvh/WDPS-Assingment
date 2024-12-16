import spacy
import argparse
import subprocess
import requests

from transformers import pipeline
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

from fact_checker import *
from entity_extractor import *
from answer_processing import *
from util import *
    
   
         
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
        
        #fetch llm output
        raw_answer = ask_question(q_text)
        raw_answer = raw_answer.lstrip(': ')
        
        if raw_answer.strip() == '':
            write_error(q_id=q_id,
                        msg='ERROR: response is empty / makes no sense',
                        path=args.outfile,
                        raw_response = raw_answer)
            continue
        
        #classify question
        expected_answer_type = classify_question(doc)

        #extract entities and disambiguate
        answer_entities = recognize_entities(text=raw_answer)
        answer_linked_entities = link_entities(entities=answer_entities)
        
        #extract entities and disambiguate
        question_entities = recognize_entities(text=q_text)
        question_linked_entities = link_entities(entities=question_entities)
        
        combined_linked_entities = question_linked_entities + answer_linked_entities
        
        #process answer
        if expected_answer_type == 'YES/NO':
            extracted_answer= extract_yes_no(raw_answer)
            
            #perform fact checking
            extracted_triplets = extract_triplets(q_text)
            fact = extract_candidate_fact(extracted_triplets, 
                                          entities = combined_linked_entities, 
                                          text = q_text)
 
            fact_true = is_property_entailed(fact['head'], fact['tail'], fact['type'])
            
            if fact_true and extracted_answer == 'yes':
                correctness = 'correct'
            else:
                correctness = 'incorrect'
                
        if expected_answer_type == 'ENTITY':
            extracted_answer_text, extracted_answer = extract_answer_entity(raw_answer, answer_linked_entities)
            
            #perform fact checking
            extracted_triplets = extract_triplets(q_text + ' ' + extracted_answer_text)
            fact = extract_candidate_fact(extracted_triplets, 
                                          entities = combined_linked_entities, 
                                          text = q_text + ' ' + extracted_answer_text)
    
            fact_true = is_property_entailed(fact['head'], fact['tail'], fact['type'])
            if fact_true:
                correctness = 'correct'
            else:
                correctness = 'incorrect'
      
        #append results to outfile
        append_outfile(path=args.outfile,
                       q_id=q_id,
                       raw_response=raw_answer,
                       answer=extracted_answer,
                       entities=answer_linked_entities,
                       correctness=correctness)
 
        
if __name__ == '__main__':
    
    main()
    