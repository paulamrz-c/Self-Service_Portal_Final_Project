# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.retriever import answer

app = FastAPI()

class Query(BaseModel):
    question: str
    model: str = "hf"

@app.post("/predict")

def predict(q: Query):
    text, category, similarity = answer(q.question, model_type=q.model)
    return {
        "answer": text,
        "category": category,
        "similarity": round(float(similarity), 2) if similarity is not None else None
    }