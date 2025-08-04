import torch
import pickle
from embedding_hf import encode_texts  # current function
from models import Classifier  
import os


MODEL_PATH = os.path.join("models", "best_query_classifier.pt")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

# load encode and model

classifier = Classifier()
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
classifier.eval()

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

def classify_query(text: str) -> str:
    """
    Classify the text 'faq', 'resource' o 'offramp' using the model.
    """ 
    embedding = encode_texts([text])  # array 2D
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

    with torch.no_grad():
        logits = classifier(embedding_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
    
    return label_encoder.inverse_transform([pred_idx])[0]

