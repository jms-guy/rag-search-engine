import os
from .keyword_search_helpers import tokenize_text
from .files import CACHE_DIR
from pickle import dump, load

INDEX_FILE = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_FILE = os.path.join(CACHE_DIR, "docmap.pkl")

class InvertedIndex:
    index: dict[str, set]
    docmap: dict[int, object]

    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    
    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), set()))
    
    def build(self, movies: list[dict]):
        for movie in movies:
            self.docmap[movie['id']] = movie

            movie_text = (f"{movie['title']} {movie['description']}")
            self.__add_document(movie['id'], movie_text)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(INDEX_FILE, 'wb') as file:
            dump(self.index, file)

        with open(DOCMAP_FILE, 'wb') as file:
            dump(self.docmap, file)

    def load(self) -> bool:
        try:
            with open(INDEX_FILE, 'rb') as file:
                self.index = load(file)
        except FileNotFoundError:
            print("Error: Index file not found")
            return True
        
        try:
            with open(DOCMAP_FILE, 'rb') as file:
                self.docmap = load(file)
        except FileNotFoundError:
            print("Error: Docmap file not found")
            return True
        
        return False