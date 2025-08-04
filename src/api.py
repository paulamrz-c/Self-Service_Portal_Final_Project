# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.retriever import answer

app = FastAPI()

class Query(BaseModel):
    question: str
    model: str = "w2v"

@app.post("/predict")
def predict(q: Query):
    result = answer(q.question, model_type=q.model)
    if "(Similarity:" in result:
        score = float(result.split("Similarity: ")[-1].replace(")", ""))
    else:
        score = 1.0
    return {"answer": result, "similarity": round(score, 2)}