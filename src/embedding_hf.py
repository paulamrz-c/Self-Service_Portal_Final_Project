from sentence_transformers import SentenceTransformer
import numpy as np


# Load the Model
model = SentenceTransformer('all-MiniLM-L6-v2')  

def encode_texts(texts: list[str]) -> np.ndarray:
    """
    Given a list of texts, return their sentence embeddings.
    Feature	                | GloVe	                               | Hugging Face (sentence-transformers)
    Vector type	            | Static per word	                   | Contextual per sentence
    Manual preprocessing	| Required (spaCy, cleaning, etc.)     | Not needed (uses its own tokenizer)
    Vector length	        | 100 (in our case)                    | ~384 (depends on the model)
    Training approach	    | Word co-occurrence matrix            | Deeper (transformers, attention mechanisms, etc.)
    Semantic precision	    | Low to medium	                       | High (better for natural language understanding)
    """
    return model.encode(texts, convert_to_numpy=True)