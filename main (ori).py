import pickle
import numpy as np
import re
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

jurnal = pickle.load(open("jurnal_list.pkl", 'rb'))
# jurnal_list = jurnal['abstract'].values

def clean_abstract(abstract):
  re.sub("[a-zA-Z ]", "", abstract)
  return abstract

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(jurnal["clean_abstract"])

with open('jurnal_list.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/jurnal-rekomen/")
def get_recommendations(abstract: str):
    abstract = clean_abstract(abstract) # Memanggil fungsi clean_abstract
    query_vec = vectorizer.transform([abstract]) # Mengubah isi dataframe abstract menjadi vektor
    similarity = cosine_similarity(query_vec, tfidf).flatten() # Perhitungan cosinus antara pencarian kata dengan dataset
    indices = np.argpartition(similarity, -5)[-5:] # Menampilkan data terdekat dari hasil pencarian, mengurutkan nilai besar ke paling depan
    sorted_indices = indices[np.argsort(similarity[indices])[::-1]]
    similarity_score = similarity[sorted_indices]
    print(similarity_score)
    results = jurnal.iloc[sorted_indices] # Menunjukan lokasi data bedasarkan indices
    json_data = results.to_json(orient='records')
    raw_json = json.loads(json_data)
    for record, value in zip(raw_json, similarity_score):
        record['skor'] = value
    # for i in sorted_indices:
    #    rekomenjurnal.append(jurnal.iloc[i].title)
    #    print(jurnal.iloc[i].title)
    # print(raw_json)
    return raw_json

get_recommendations('Corona')


    # #print(TextInput)
    # query_vec = vectorizer.transform([TextInput]) # Mengubah isi dataframe title menjadi vektor
    
    # similarity = cosine_similarity(query_vec, tfidf).flatten() # Perhitungan cosinus antara pencarian kata dengan dataset
    # indices = np.argpartition(similarity, -5)[-5:] # Mendapatkan 5 data terdekat dari hasil pencarian
    # results = jurnal.iloc[indices].iloc[::-1] # Menunjukan lokasi data bedasarkan indices
    # #json_data = results.to_json()
    # rekomenjurnal = []
    # print(rekomenjurnal)
    # for journalId in results.journalId:
    #    rekomenjurnal.append(jurnal.iloc[journalId].title)
       #print(jurnal.iloc[journalId].title)
    #print(jurnal)
    #print(results)
    # print(rekomenjurnal)
    # return rekomenjurnal
    


#jurnal_rekomen('Cancer and cure: A critical analysis')