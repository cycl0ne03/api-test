import pandas as pd
import nltk
import nltk.tokenize
import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jurnal = pd.read_csv(r"datajurnal2.csv",sep=";")

def clean_title(title):
  re.sub("[^a-zA-Z0-9 ]", "", title)
  return title

def clean_abstract(abstract):
  re.sub("[a-zA-Z ]", "", abstract)
  return abstract

jurnal["clean_title"] = jurnal["title"].apply(clean_title)
jurnal["clean_abstract"] = jurnal["abstract"].apply(clean_abstract)
jurnal["combined"] = jurnal["clean_title"] + " " + jurnal["clean_abstract"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(jurnal["combined"])

cosine_sim_matrix = cosine_similarity(tfidf_matrix)

def get_recommendations(cosine_sim_matrix, titles, user_input, top_n=5):
    user_tfidf = vectorizer.transform([user_input])
    user_cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    
    sim_scores = list(enumerate(user_cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[:top_n]

    first_indices = [item[0] for item in sim_scores]
    second_indices = [item[1] for item in sim_scores]

    results = jurnal.iloc[first_indices] 

    json_data = results.to_json(orient='records')
    raw_json = json.loads(json_data)

    for record, value in zip(raw_json, second_indices):
        record['skor'] = f"{value*100:.2f}%"

    print(raw_json)

    return raw_json

model_data = {
    'vectorizer': vectorizer,
    'cosine_similarity': cosine_sim_matrix,
    'documents': jurnal["clean_abstract"],
    'titles': jurnal["clean_title"],
    'get_recommendations': get_recommendations
}

with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)