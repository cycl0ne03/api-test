import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from journal_model import get_recommendations

with open('recommendation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

loaded_titles = model_data['titles']
loaded_cosine_sim = model_data['cosine_similarity']
get_recommendations = model_data['get_recommendations']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/jurnal-rekomen/")
async def recommend(input: str):
    recommendations = get_recommendations(loaded_cosine_sim, loaded_titles, input)
    # print(recommendations)
    return recommendations
