import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import pickle
import os

def load_relations():
    relations_df = pd.read_excel('wikidata_relation_types.xlsx')
    relations_df = relations_df[relations_df['count'] > 3]
    print(len(relations_df), "relations loaded.")
    relations_df = relations_df[['relation_label', 'relation_description']]
    relations_df['relation_text'] = relations_df['relation_label'].fillna('') + ' - ' + relations_df['relation_description'].fillna('')
    return relations_df

def compute_relation_embeddings(embed_model, relation_descriptions, cache_path='relation_embeddings.pkl'):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            relation_embeddings = pickle.load(f)
        print("Loaded cached embeddings.")
    else:
        relation_embeddings = embed_model.encode(relation_descriptions, convert_to_tensor=True, show_progress_bar=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(relation_embeddings, f)
        print("Computed and cached embeddings.")
    return relation_embeddings

def find_relation(embed_model, zero_shot, relation_embeddings, relation_labels, e1, e2, context, TOP_N):
    context_emb = embed_model.encode(context, convert_to_tensor=True)
    cos_scores = util.cos_sim(context_emb, relation_embeddings)[0]
    top_indices = torch.topk(cos_scores, k=TOP_N).indices
    candidate_labels = [relation_labels[i] for i in top_indices]
    result = zero_shot(sequences=context, candidate_labels=candidate_labels, multi_label=False)
    top_relation, top_score = result['labels'][0], result['scores'][0]
    return top_relation, top_score

def process_texts(texts, entities_list, embed_model, zero_shot, relation_embeddings, relation_labels, TOP_N):
    for text, entities in zip(texts, entities_list):
        if len(entities) < 2:
            print(f"Text: '{text}'\nNo entity pairs found.\n")
            continue
        print(f"Text: '{text}'")
        for i in range(len(entities) - 1):
            e1, e2 = entities[i], entities[i + 1]
            top_relation, top_score = find_relation(embed_model, zero_shot, relation_embeddings, relation_labels, e1, e2, text, TOP_N)
            print(f"  Entities: {e1} - {e2} -> Predicted Relation: {top_relation} (Score: {top_score:.4f})")
        print("\n")

# Initialization
relations_df = load_relations()
relation_labels = relations_df['relation_label'].tolist()
relation_descriptions = relations_df['relation_description'].tolist()
embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
relation_embeddings = compute_relation_embeddings(embed_model, relation_descriptions)
zero_shot = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
TOP_N = 100

# Example usage
texts = [
    "Is it correct that Berlin is the capital of Germany?",
    "Which island is known for its Moai statues?",
    "Does Japan have the highest life expectancy in the world?",
    "True or False: Brazil’s official language is Portuguese.",
    "Would it be accurate to say the Eiffel Tower is in Paris?",
    "Who was the first Nobel Prize winner?",
    "Is the Amazon Rainforest located in Brazil?",
    "Confirm or Deny: Canada is the second-largest country by area.",
    "Would it be true to say that Tokyo is the largest city in Japan?",
    "Is the Great Wall of China located in China?",
    "The Great Wall of China is visible from space: Correct or Incorrect?",
    "Is it true that Mount Everest is the tallest mountain on Earth?",
    "Verify if Antarctica is the coldest continent.",
    "Paris is the capital of France. Yes or No?",
    "Is New York City the capital of the United States?",
    "Should we consider the Amazon River to flow through Brazil?",
    "Is it correct that Canada has ten provinces?",
    "Does the Sahara Desert lie in South America?",
    "Who painted the Mona Lisa?",
    "Verify if the Pacific Ocean is the largest ocean on Earth.",
    "Can it be said that the Nile River flows through Egypt?",
    "The Statue of Liberty is located in France: True or False?",
    "Is it the case that Norway is part of Scandinavia?",
    "Is Ronaldo the top scorer of Real Madrid?",
    "Is Brazil in South America?",
    "Is the Leaning Tower located in Pisa, Italy?",
    "Is it correct to say that Venus is closer to the Sun than Earth?",
    "Does Mount Fuji lie in Japan?",
    "What is the capital of Japan?",
    "Should we say that the English Channel separates England and France?",
    "Who is the founder of Microsoft?",
    "Is the Dead Sea one of the world’s saltiest bodies of water?",
    "True or False: Rome is the capital of Italy.",
    "Is South Africa the only country with three capital cities?",
    "Does Mexico share a border with Canada?",
    "Is Sydney the capital of Australia?",
    "The Pyramids of Giza are in Egypt. True or False?",
    "Which planet is known as the Red Planet?",
    "Confirm or Deny: Pluto is no longer considered a planet.",
    "Where is the Great Wall located?",
    "The largest desert on Earth is the Sahara. Yes or No?",
    "Name the scientist who developed the theory of relativity.",
    "What country is known for inventing pizza?",
    "What is the largest country in South America?",
    "Who wrote the novel Pride and Prejudice?",
    "Where can you find the Leaning Tower?",
    "Who directed the movie Inception?",
    "What is the longest river in the world?",
    "Name the tallest mountain on Earth.",
    "Which city is known as the Big Apple?",
    "Who is known as the Father of Modern Physics?",
    "What ocean lies between Africa and Australia?",
    "Identify the capital city of Canada.",
    "Which country is the origin of sushi?",
    "Who was the first person to walk on the moon?",
    "What is the main language spoken in Brazil?",
    "Where can you find the Taj Mahal?",
    "Who painted Starry Night?",
    "Which river flows through London?",
    "Who was the first president of the United States?",
    "What country has the longest coastline?",
    "Who is the CEO of Tesla?",
    "What city is home to the Colosseum?",
    "Which chemical element has the symbol O?",
    "Name the largest desert in the world.",
    "Who is the current monarch of England?",
    "What country borders the United States to the north?",
    "Where is the world’s largest coral reef?",
    "What is the capital of Egypt?",
    "Who discovered penicillin?",
    "Identify the country with the most spoken language, Mandarin.",
    "What animal is known as the King of the Jungle?",
    "Which continent is the Sahara Desert on?",
    "Name the founder of Facebook.",
    "Who is the author of The Odyssey?"
]

entities_list = [
    ["Berlin", "Germany"],
    ["Moai statues", "island"],
    ["Japan", "highest life expectancy"],
    ["Brazil", "Portuguese"],
    ["Eiffel Tower", "Paris"],
    ["first Nobel Prize winner"],
    ["Amazon Rainforest", "Brazil"],
    ["Canada", "second-largest country"],
    ["Tokyo", "largest city in Japan"],
    ["Great Wall of China", "China"],
    ["Great Wall of China", "space"],
    ["Mount Everest", "tallest mountain"],
    ["Antarctica", "coldest continent"],
    ["Paris", "France"],
    ["New York City", "capital of the United States"],
    ["Amazon River", "Brazil"],
    ["Canada", "ten provinces"],
    ["Sahara Desert", "South America"],
    ["Mona Lisa", "painter"],
    ["Pacific Ocean", "largest ocean"],
    ["Nile River", "Egypt"],
    ["Statue of Liberty", "France"],
    ["Norway", "Scandinavia"],
    ["Ronaldo", "top scorer of Real Madrid"],
    ["Brazil", "South America"],
    ["Leaning Tower", "Pisa", "Italy"],
    ["Venus", "closer to the Sun", "Earth"],
    ["Mount Fuji", "Japan"],
    ["capital of Japan"],
    ["English Channel", "England", "France"],
    ["founder of Microsoft"],
    ["Dead Sea", "saltiest bodies of water"],
    ["Rome", "capital of Italy"],
    ["South Africa", "three capital cities"],
    ["Mexico", "Canada"],
    ["Sydney", "capital of Australia"],
    ["Pyramids of Giza", "Egypt"],
    ["Red Planet"],
    ["Pluto", "planet"],
    ["Great Wall"],
    ["Sahara", "largest desert"],
    ["scientist", "theory of relativity"],
    ["country", "inventing pizza"],
    ["largest country in South America"],
    ["author", "Pride and Prejudice"],
    ["Leaning Tower"],
    ["director", "Inception"],
    ["longest river"],
    ["tallest mountain"],
    ["city", "Big Apple"],
    ["Father of Modern Physics"],
    ["ocean", "Africa", "Australia"],
    ["capital city of Canada"],
    ["country", "origin of sushi"],
    ["first person", "walk on the moon"],
    ["main language", "Brazil"],
    ["Taj Mahal"],
    ["painter", "Starry Night"],
    ["river", "London"],
    ["first president of the United States"],
    ["country", "longest coastline"],
    ["CEO of Tesla"],
    ["city", "Colosseum"],
    ["chemical element", "symbol O"],
    ["largest desert"],
    ["current monarch of England"],
    ["country", "borders the United States to the north"],
    ["world’s largest coral reef"],
    ["capital of Egypt"],
    ["discovered penicillin"],
    ["country", "most spoken language", "Mandarin"],
    ["animal", "King of the Jungle"],
    ["continent", "Sahara Desert"],
    ["founder of Facebook"],
    ["author", "The Odyssey"]
]

process_texts(texts, entities_list, embed_model, zero_shot, relation_embeddings, relation_labels, TOP_N)