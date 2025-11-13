import numpy as np
import os
from sentence_transformers import SentenceTransformer
from .files import CACHE_DIR

EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    model: SentenceTransformer
    embeddings: np.ndarray
    documents: list[dict]

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        self.embeddings = None 
        self.documents = None 
        self.document_map = {}

    def generate_embedding(self, text: list[str]) -> np.ndarray:
        embeddings = self.model.encode(sentences=text, show_progress_bar=True)
        self.embeddings = embeddings

        np.save(EMBEDDINGS_CACHE, self.embeddings)
        return self.embeddings
    
    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}


        movies = []
        for doc in documents:     
            movie_data = f"{doc['title']}: {doc['description']}"
            movies.append(movie_data)
        
        self.generate_embedding(movies)

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        if not os.path.exists(EMBEDDINGS_CACHE):
            self.embeddings = self.build_embeddings(documents)
            return self.embeddings
        
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        self.embeddings = np.load(EMBEDDINGS_CACHE)
        if len(self.embeddings) == len(documents):
            return self.embeddings
        else:
            return self.build_embeddings(documents)

def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")
