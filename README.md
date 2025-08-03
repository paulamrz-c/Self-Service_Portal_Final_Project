# Student Self-Service Chatbot: Proof of Concept

This repository contains a Proof of Concept (PoC) for a chatbot that answers frequently asked questions and connects students with resources using NLP and semantic similarity.

## Project Structure

Self-Service_Portal_Final_Project/
├── data/
│   ├── raw/                        # Original PDFs and scraped CSV
│   └── processed/                 # Cleaned FAQs, resources, and embeddings
├── models/
│   └── embeddings/                # Trained Word2Vec model + downloaded GloVe
├── notebooks/
│   └── 01_scrapping.ipynb         # Scrapping Success Page
│   └── 02_extract_FAQs.ipynb      # Extraction of FAQs from PDFs
│   └── 03_build_embeddings.ipynb  # Preprocessing and vector training
├── src/
│   ├── retriever.py               # Core logic to return relevant answers
│   └── chatbot_interface.py       # Streamlit chatbot interface
│   └── test.py                    # Local test of chatbot (optional)
└── README.md                      # You're here!

## NLP Pipeline

We used a lightweight but effective NLP pipeline:

- ftfy, unicodedata, re: fix encoding and normalize text
- spaCy: tokenization + lemmatization
- stopword removal: remove unhelpful filler words
- We applied this pipeline both to our corpus (FAQs + resources) and to incoming user questions.

## Corpus Construction

- FAQs were extracted from Winter 2024 Student Fees and Registrar PDF documents.
- Resources were scraped from the Conestoga Student Success Portal and cleaned.
- Both were merged into a unified corpus and vectorized.

## Each entry includes:

text: question/title + answer/description
source: faq or resource
payload: final answer or link shown to the user

### Embeddings

Two types of embeddings were used:

🔹 Word2Vec (custom) --> Trained on our actual student data (FAQs + resources)

Captures relationships specific to Conestoga context

🔹 GloVe (pre-trained) --> 100-dimensional GloVe.6B vectors from Stanford NLP

Brings external general knowledge

### Vectors were saved in:

data/processed/glove_vectors.pkl

data/processed/word2vec_vectors.pkl

## Retriever Logic

Implemented in retriever.py:

- Preprocess the user query
- Vectorize it using the selected model
- Compare it with all document vectors using cosine similarity
- Return the most similar answer (payload) if similarity > 0.4

If not, return fallback message to escalate

## Streamlit Interface

A simple app in src/app.py lets you test the chatbot:

- streamlit run src/app.py

### Features:

- Input field for question
- Option to select embedding model (Word2Vec or GloVe)
- Returns matched response and its source

## Next Steps

Fine-tune similarity threshold
Add UI elements for feedback or human escalation
Deploy on public link (Streamlit Cloud, Render, etc.)

👥 Authors

- Babandeep 9001552
- Hasyashri Bhatt 9028501
- Paula Ramirez 8963215

📄 License

This is a student PoC project. Not intended for production use without privacy and data compliance review.