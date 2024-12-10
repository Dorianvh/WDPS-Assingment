import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

def load_relations():
    relations_df = pd.read_excel('wikidata_relation_types.xlsx')
    relations_df = relations_df[relations_df['count'] > 3]
    print(len(relations_df), "relations loaded.")
    relations_df = relations_df[['relation_label', 'relation_description']]
    relations_df['relation_text'] = relations_df['relation_label'].fillna('') + ' - ' + relations_df['relation_description'].fillna('')
    return relations_df

def compute_relation_embeddings(embed_model, relation_descriptions):
    return embed_model.encode(relation_descriptions, convert_to_tensor=True, show_progress_bar=True)

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
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
relation_embeddings = compute_relation_embeddings(embed_model, relation_descriptions)
zero_shot = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
TOP_N = 100

# Example usage
texts = [
    "Barack Obama was born in Honolulu, Hawaii.",
    "Apple announced the new iPhone in Cupertino.",
    "Elon Musk founded SpaceX in 2002."
]

entities_list = [
    ["Barack Obama", "Honolulu", "Hawaii"],
    ["Apple", "iPhone", "Cupertino"],
    ["Elon Musk", "SpaceX", "2002"]
]

process_texts(texts, entities_list, embed_model, zero_shot, relation_embeddings, relation_labels, TOP_N)

