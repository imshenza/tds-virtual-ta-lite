# api/main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
import sqlite3
import numpy as np
import json
import os
import requests

app = FastAPI()

# Load all embeddings from the precomputed SQLite DB
conn = sqlite3.connect("data/embeddings.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("SELECT id, content, source, embedding FROM embeddings")
rows = cursor.fetchall()

# Cache embeddings in memory
EMBEDDINGS = []
for id_, content, source, emb_blob in rows:
    emb = np.frombuffer(emb_blob, dtype=np.float32)
    EMBEDDINGS.append((id_, content, source, emb))

class QueryRequest(BaseModel):
    question: str

# Use OpenRouter for generation (or replace with any other lightweight API)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://your-app-name.onrender.com",
    "X-Title": "TDS Virtual TA"
}

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/query")
def query(request: QueryRequest):
    # Load embedding model only for single text
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight

    question_emb = model.encode(request.question)

    # Find top 3 most similar chunks
    similarities = [(id_, content, source, cosine_similarity(question_emb, emb)) for id_, content, source, emb in EMBEDDINGS]
    top_chunks = sorted(similarities, key=lambda x: x[3], reverse=True)[:3]
    context = "\n\n".join([chunk[1] for chunk in top_chunks])

    # Generate answer using OpenRouter
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful teaching assistant."},
            {"role": "user", "content": f"Answer this based on context:\n\n{context}\n\nQuestion: {request.question}"}
        ]
    }
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=HEADERS, json=payload)
    answer = res.json()["choices"][0]["message"]["content"]

    # Include links
    links = list(set(chunk[2] for chunk in top_chunks))

    return {"answer": answer, "links": links}
