""" 
=======================
This File:
=======================

In this file we will create the retriever, which is the component that, given a user input text, 
searches for the most relevant response within the database we built previously.

1. User types a question → “What is VMock?” (Loads vectors (.pkl) and original texts)
2. The retriever converts that question into a vector.
3. It compares that vector with all the vectors in the corpus (FAQs + resources).
4. It returns the most similar vector (by cosine similarity).

"""

import pandas as pd
import numpy as np
import spacy
import ftfy, unicodedata, re
from sklearn.metrics.pairwise import cosine_similarity
import pickle

### =======================
### Load NLP model (spaCy)
### =======================

# Load spaCy model with tagger only (for lemmatization)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

### =======================
### Text preprocessing - Same that corpus
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
    Convert text into a sentence vector by averaging word embeddings.
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

# Load precomputed vectors (Word2Vec or GloVe)
vectors_glove = pd.read_pickle("./data/processed/glove_vectors.pkl")["vec_glove"].tolist()
vectors_w2v   = pd.read_pickle("./data/processed/word2vec_vectors.pkl")["vec_w2v"].tolist()

### =======================
### Answer function
### =======================

def answer(query: str, model_type="glove", top_k=1) -> str:
    """
    Given a query string, return the most relevant answer
    from the corpus using either Word2Vec or GloVe embeddings.
    """
    if model_type == "glove":
        from gensim.models import KeyedVectors
        glove_path = "./models/embeddings/glove.6B.100d.txt"
        model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
        vecs = vectors_glove
    elif model_type == "w2v":
        from gensim.models import Word2Vec
        model = Word2Vec.load("./models/embeddings/word2vec_faqs.bin").wv
        vecs = vectors_w2v
    else:
        raise ValueError("model_type must be 'glove' or 'w2v'")

    # Vectorize user query
    query_vec = sent_vector(query, model, dim=100).reshape(1, -1)

    # Calculate cosine similarity
    matrix = np.vstack(vecs)
    sims = cosine_similarity(query_vec, matrix)[0]

    # Get top result(s)
    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]
    best_doc = docs.iloc[best_idx]

    # Optional off-ramp (if similarity is too low)
    if best_sim < 0.4:
        return f"🤖 I'm not sure how to help with that. Please contact an advisor.\n(Similarity: {best_sim:.2f})"

    return f"{best_doc['payload']}  \n\n[Source: {best_doc['source']}]  \n(Similarity: {best_sim:.2f})"
