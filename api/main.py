import os
import sqlite3
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import requests
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

DB_PATH = "data/embeddings.db"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class Query(BaseModel):
    question: str
    image: Optional[str] = None

app = FastAPI()

def get_top_chunks(query, k=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, source, embedding FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    contents = [row[1] for row in rows]
    sources = [row[2] for row in rows]
    embeddings = [row[3] for row in rows]

    import numpy as np
    import torch

    query_emb = MODEL.encode(query, convert_to_numpy=True)
    doc_embs = np.array([np.frombuffer(e, dtype=np.float32) for e in embeddings])
    scores = util.cos_sim(torch.tensor(query_emb), torch.tensor(doc_embs))[0]

    top_k_idx = torch.topk(scores, k).indices.tolist()
    return [{"content": contents[i], "source": sources[i]} for i in top_k_idx]

def generate_answer(query, chunks):
    context = "\n\n".join([chunk["content"] for chunk in chunks])
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You're a helpful teaching assistant for the IIT Madras Data Science program."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error generating answer: {e}"

@app.post("/query")
def handle_query(q: Query):
    chunks = get_top_chunks(q.question)
    answer = generate_answer(q.question, chunks)
    links = list(set(chunk["source"] for chunk in chunks))
    return {"answer": answer, "links": links}
