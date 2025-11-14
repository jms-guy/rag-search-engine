import numpy as np
import os
from sentence_transformers import SentenceTransformer
from .files import CACHE_DIR
from .index import _load_pickle, DOCMAP_FILE

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
        self.load_for_search()
        print(f"Embeddings length: {len(self.embeddings)}")
        print(f"Docmap length: {len(self.document_map)}")

        query_embedding = self.generate_embedding(query)
        similarity_scores = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.embeddings]
        similarities = []
        print(f"Similarity score length: {len(similarity_scores)}")

        for i in range(len(self.document_map)):
            similarities.append((similarity_scores[i], self.document_map[i]))

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
    
    def load_for_search(self):
        try:
            self.embeddings = np.load(EMBEDDINGS_CACHE)
        except:
            raise ValueError("No embeddings loaded. Call 'load_or_create_embeddings' first.")
        
        try:
            self.document_map = _load_pickle(DOCMAP_FILE)
        except FileNotFoundError:
            print("Docmap file not found")

def verify_model():
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)