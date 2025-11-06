import os
from .keyword_search_helpers import tokenize_text
from .files import CACHE_DIR
from pickle import dump, load
from collections import Counter

INDEX_FILE = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_FILE = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_FILE = os.path.join(CACHE_DIR, "term_frequencies.pkl")

class InvertedIndex:
    index: dict[str, set]
    docmap: dict[int, object]
    term_frequencies: dict[int, Counter]

    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
    
    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), set()))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        token_terms = tokenize_text(term)
        if len(token_terms) > 1:
            raise Exception("Expecting single token for get_tf function")
        token = token_terms[0]
        counter = self.term_frequencies.get(doc_id, Counter())
        return counter[token]
    
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

        with open(TERM_FREQUENCIES_FILE, 'wb') as file:
            dump(self.term_frequencies, file)

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
        
        try:
            with open(TERM_FREQUENCIES_FILE, 'rb') as file:
                self.term_frequencies = load(file)
        except FileNotFoundError:
            print("Error: Term Frequencies file not found")
            return True
        
        return False