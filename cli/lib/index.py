import os
import math
from .keyword_search_helpers import tokenize_text
from .files import CACHE_DIR
from pickle import dump, load
from collections import Counter

INDEX_FILE = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_FILE = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_FILE = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTHS_FILE = os.path.join(CACHE_DIR, "doc_lengths.pkl")

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    index: dict[str, set]
    docmap: dict[int, object]
    term_frequencies: dict[int, Counter]
    doc_lengths: dict[int, int]

    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

    def _get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return avg_doc_length
    
    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), set()))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        token_terms = tokenize_text(term)
        if len(token_terms) > 1:
            raise Exception("Expecting single token for get_tf function")
        token = token_terms[0]
        counter = self.term_frequencies.get(doc_id, Counter())
        return counter[token]
    
    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
                raise Exception("Expecting single token for bm25_idf function")
        term = tokens[0]

        number_of_docs = len(self.docmap)
        term_doc_set = self.index.get(term, set())
        term_doc_count = len(term_doc_set)

        bm25_idf = math.log((number_of_docs - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
        return bm25_idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1 = BM25_K1, b = BM25_B) -> float:
        avg_doc_length = self._get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)

        tf = self.get_tf(doc_id, term)
        normalized_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return normalized_tf
    
    def get_bm25(self, doc_id: int, term: str) -> float:
        bm25tf = self.get_bm25_tf(doc_id, term, BM25_K1, BM25_B)
        bm25idf = self.get_bm25_idf(term)

        return bm25tf * bm25idf
    
    def bm25_search(self, query: str, limit: int) -> list:
        query_tokens = tokenize_text(query)
        scores = {}
        relevant_docs = set()

        for token in query_tokens:
            token_docs = self.index.get(token)
            if token_docs == None:
                continue
            relevant_docs.update(token_docs)

        for doc in relevant_docs:
            scores[doc] = 0.0
            for token in query_tokens:
                if token not in self.index:
                    continue
                if doc not in self.index[token]:
                    continue
                token_bm25 = self.get_bm25(doc, token)
                scores[doc] += token_bm25
        
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sliced_scores = sorted_scores[:limit]

        return sliced_scores

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
        
        with open(DOC_LENGTHS_FILE, 'wb') as file:
            dump(self.doc_lengths, file)

    def load(self) -> bool:
        try:
            self.index = _load_pickle(INDEX_FILE)
            self.docmap = _load_pickle(DOCMAP_FILE)
            self.term_frequencies = _load_pickle(TERM_FREQUENCIES_FILE)
            self.doc_lengths = _load_pickle(DOC_LENGTHS_FILE)
            return True
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False
    
def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return load(f)
    except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing required file: {path}") from e