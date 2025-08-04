""" 
=======================
This File:
=======================

This file implements the "retriever" component of the chatbot system. Given a student query, 
it searches for the most semantically relevant answer within the database built previously 
from FAQs and student resources.

Process:
1. User types a question → e.g., “What is VMock?”
2. The retriever converts that question into a vector, using one of:
   - GloVe (static word embeddings)
   - Word2Vec (trained on the local corpus)
   - Hugging Face Sentence Transformers (contextual sentence embeddings)
3. It compares the query vector to all stored vectors in the corpus using cosine similarity.
4. It returns the most similar entry (FAQ or resource) or escalates if similarity is too low.

"""


import pandas as pd
import numpy as np
import spacy
import ftfy, unicodedata, re
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
import os
from transformers import pipeline
sys.path.append(os.path.dirname(__file__))
from query_classifier import classify_query


from embedding_hf import encode_texts
### =======================
### Load NLP model (spaCy)
### =======================

# Load spaCy model with tagger only (for lemmatization)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

### =======================
### Text preprocessing - Same that corpus (used only for GloVe/W2V)
### =======================

def preprocess(text: str) -> list[str]:
    """
    Clean and tokenize input text, returning lemmatized tokens.
    """
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-–—]", "-", text)
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]

def sent_vector(text, model, dim=100):
    """
    Convert text into a sentence vector by averaging word embeddings (GloVe/Word2Vec only).
    """
    tokens = preprocess(text)
    vecs = [model[t] for t in tokens if t in model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

### =======================
### Load embeddings + corpus
### =======================

# Load the preprocessed corpus (FAQ + resources)
corpus_path = "./data/processed/faqs.csv"
res_path = "./data/processed/student_resources_index.csv"

faq_df = pd.read_csv(corpus_path)
faq_df["text"] = faq_df["question"].astype(str)
faq_df["source"] = "faq"
faq_df["payload"] = faq_df["answer"]

try:
    res_df = pd.read_csv(res_path)
    res_df["text"] = res_df["title"].astype(str) + " " + res_df["description"].astype(str)
    res_df["source"] = "resource"
    res_df["payload"] = res_df["description"] + "\n🔗 " + res_df["link"].astype(str)
except:
    res_df = pd.DataFrame(columns=["text", "source", "payload"])

# Combine all documents
docs = pd.concat([faq_df[["text", "source", "payload"]],
                  res_df[["text", "source", "payload"]]],
                 ignore_index=True)

# Load precomputed vectors (Word2Vec, GloVe, Hugging Face)
vectors_glove = pd.read_pickle("./data/processed/glove_vectors.pkl")["vec_glove"].tolist()
vectors_w2v   = pd.read_pickle("./data/processed/word2vec_vectors.pkl")["vec_w2v"].tolist()
with open("./data/processed/hf_vectors.pkl", "rb") as f:
    vectors_hf = pickle.load(f)


### =======================
### Load Transformer and generate function
### =======================

generator = pipeline("text-generation", model="distilgpt2")

def generate_llm_response(query):
    prompt = (
        "You are a Student Success Advisor at Conestoga College. "
        "If the student says hello, greet them politely. "
        "Keep your answers short and friendly.\n"
        f"Student: {query}\nAdvisor:"
    )
    response = generator(prompt, max_length=60, num_return_sequences=1, do_sample=True, temperature=0.8)[0]["generated_text"]
    
    # Extraer solo lo generado después de "Advisor:"
    if "Advisor:" in response:
        return response.split("Advisor:")[-1].strip().split("\n")[0]
    else:
        return response.strip()

### =======================
### Answer function
### =======================

def answer(query: str, model_type="hf", top_k=1) -> tuple[str, str, float | None]:
    """
    Given a query string, return the most relevant answer
    from the corpus using either Word2Vec, GloVe or Hugging Face embeddings.
    """
    category = classify_query(query)
    print(f"🧠 CLASSIFIED AS: {category} for: {query}")    

    if category == "chitchat":
        # ✅ Generar con LLM solo si es chitchat
        llm_response = generate_llm_response(query)
        return (
            f"{llm_response}\n\n_(Generated using LLM for chitchat)_",
            category,
            None,
        )
    
    if category == "offramp":
        return (
            "🤖 This question may require human support. We've flagged it for a student advisor.",
            category,
            None,
        )

    if model_type == "glove":
        from gensim.models import KeyedVectors
        glove_path = "./models/embeddings/glove.6B.100d.txt"
        model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
        vecs = vectors_glove
        query_vec = sent_vector(query, model, dim=100).reshape(1, -1)

    elif model_type == "w2v":
        from gensim.models import Word2Vec
        model = Word2Vec.load("./models/embeddings/word2vec_faqs.bin").wv
        vecs = vectors_w2v
        query_vec = sent_vector(query, model, dim=100).reshape(1, -1)

    elif model_type == "hf":
        # === Hugging Face embedding (sentence-transformers) ===        
        vecs = vectors_hf
        query_vec = encode_texts([query])  # Already returns 2D array

    else:
        raise ValueError("model_type must be 'glove', 'w2v' or 'hf'")

    # Calculate cosine similarity
    matrix = np.vstack(vecs)
    sims = cosine_similarity(query_vec, matrix)[0]

    # Get top result(s)
    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]
    best_doc = docs.iloc[best_idx]


    # Optional off-ramp (if similarity is too low)
    if best_sim < 0.4:
        llm_response = generate_llm_response(query)
        return (
            f"{llm_response}\n\n_(Generated using LLM fallback)_",
            category,
            best_sim
        )

    return (
        f"{best_doc['payload']}  \n\n[Generated using: {best_doc['source']}]  \n(Similarity: {best_sim:.2f})",
        category,
        best_sim,
    )