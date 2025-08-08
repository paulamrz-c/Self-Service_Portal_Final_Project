# Student Self-Service Chatbot: Proof of Concept

This repository presents a **Proof of Concept (PoC)** for a chatbot designed to reduce the workload of Student Success Advisors by answering frequent student queries using **NLP**, **semantic search**, and **predictive classification**.

## Project Structure

Self-Service_Portal_Final_Project/

├── data/ # Raw & processed FAQs/resources & embeddings

├── models/ # Trained classifier, Word2Vec, GloVe

├── notebooks/ # Jupyter notebooks for data processing

├── src/ # All source code

│ ├── api.py # FastAPI app for predictions

│ ├── chatbot_interface.py # Streamlit frontend

│ ├── retriever.py # Core logic (semantic search + classifier)

│ ├── query_classifier.py # PyTorch classifier

│ └── embedding_hf.py # SentenceTransformer embeddings

├── requirements.txt # Python dependencies

└── README.md # This file

## NLP Pipeline

-  Text normalization: `ftfy`, `unicodedata`, `re`
-  Lemmatization & tokenization: `spaCy`
-  Vectorization: Word2Vec, GloVe, and HuggingFace Sentence Transformers
-  Query classification: `faq`, `resource`, `chitchat`, `offramp` (using PyTorch model)

## Corpus Construction

- Extracted FAQs from Winter 2024 PDF documents
- Scraped resources from Conestoga's [Student Success Portal](https://successportal.conestogac.on.ca/)
- Combined into a unified document base with associated embeddings


##  Retrieval Logic (Retriever)

Implemented in `src/retriever.py`:
- Classify the query
- If similarity > 0.9 (even for `chitchat`), override and return the matched answer
- Otherwise, fall back to a generative model (`distilgpt2`) or escalate to human advisor


##  Demo Features
Ask natural language questions like:
"Where can I upload my ONE Card photo?"
"Where can I find my timetable?"
"What is VMock?"
"How can I pay my fees"

### Answers are returned with:
💬 Relevant info (answer or link)
📄 Source (FAQ or resource)
📈 Similarity score
🔁 Escalation fallback via LLM (if needed)

## Models Used
- Word2Vec: Trained on internal FAQs + resources
- GloVe: Pre-trained 100d vectors from Stanford NLP
- HuggingFace: all-MiniLM-L6-v2 for contextual sentence embeddings
- DistilGPT2: Used as LLM fallback for chitchat and low similarity

## To Do
- Override chitchat if semantic similarity is very high
- Feedback loop: Let users flag incorrect answers
- Deploy to Streamlit Cloud / Render

## Authors
- Paula Ramirez (8963215)
- Babandeep (9001552)
- Hasyashri Bhatt (9028501)


##  **How to Run the Application**

### 1. Activate your virtual environment (PowerShell):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\activate
```
### 2. Run the FastAPI backend:
```powershell
streamlit run src/chatbot_interface.py --server.port 8501
```
### 3. Launch the Streamlit chatbot interface:
```powershell
streamlit run src/chatbot_interface.py --server.port 8501
```

