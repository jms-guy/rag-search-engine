import numpy as np
import os
from sentence_transformers import SentenceTransformer
from .files import CACHE_DIR
from .semantic_search_helpers import cosine_similarity

EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    model: SentenceTransformer
    embeddings: np.ndarray
    documents: list[dict]
    document_map: dict[int, object]

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        self.embeddings = None 
        self.documents = None 
        self.document_map = {}

    def search(self, query: str, limit: int) -> list[dict]:
        if not os.path.exists(EMBEDDINGS_CACHE):
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        self.embeddings = np.load(EMBEDDINGS_CACHE)

        query_embedding = self.generate_embedding(query)
        similarity_scores = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.embeddings]

        for i in range(len(self.document_map)):
            similarities = [(score, self.document_map[i]) for score in similarity_scores]

        sorted_scores = sorted(similarities, reverse=True)
        sorted_scores = sorted_scores[:limit]

        results = [dict]
        for score, obj in sorted_scores:
            result = {'score': score, 'title': obj['title'], 'description': obj['description']}
            results.append(result)

        return results

    def generate_embedding(self, text: list[str]) -> np.ndarray:
        self.embeddings = self.model.encode(sentences=text, show_progress_bar=True)
        return self.embeddings
    
    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        movies = [f"{doc['title']}: {doc['description']}" for doc in documents]
        embeddings = self.generate_embedding(movies)
        np.save(EMBEDDINGS_CACHE, embeddings)

        return embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if os.path.exists(EMBEDDINGS_CACHE):
            self.embeddings = np.load(EMBEDDINGS_CACHE)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")
