import os
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "data/embeddings.db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI()

class Query(BaseModel):
    question: str
    image: Optional[str] = None

def search_chunks(query, k=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, source FROM embeddings")
    rows = cursor.fetchall()
    conn.close()

    # Use naive keyword ranking instead of model-based similarity
    ranked = sorted(rows, key=lambda row: query.lower() in row[1].lower(), reverse=True)
    top_chunks = [{"id": r[0], "content": r[1], "source": r[2]} for r in ranked[:k]]
    return top_chunks

def generate_answer_openrouter(query, context_chunks):
    context_text = "\n\n".join([chunk['content'] for chunk in context_chunks])
    prompt = f"""Answer the question based on the following context:\n\n{context_text}\n\nQuestion: {query}"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You're a helpful TA for IIT Madras Data Science."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Failed to get response from GPT: {str(e)}"

@app.post("/query")
def query_route(query: Query):
    top_chunks = search_chunks(query.question, k=5)
    answer = generate_answer_openrouter(query.question, top_chunks)
    links = list(set(chunk["source"] for chunk in top_chunks))
    return {"answer": answer, "links": links}
