import math
from .index import InvertedIndex
from .keyword_search_helpers import tokenize_text

DEFAULT_SEARCH_LIMIT = 5

def search_command(index: InvertedIndex, query: str) -> list[object]:
    doc_ids = find_matches(index, query)
    movies = find_documents(index, doc_ids)
    return movies

def find_matches(index: InvertedIndex, query: str) -> set[int]:
    results = set()

    query_tokens = tokenize_text(query)
    for token in query_tokens:
        doc_ids = index.get_documents(token)

        for id in doc_ids:
            if len(results) >= 5:
                return results
            results.add(id)
    return results

def find_documents(index: InvertedIndex, doc_ids: set[int]) -> list[object]:
    movies = []

    for id in doc_ids:
        movies.append(index.docmap[id])

    return movies

def calc_idf(index: InvertedIndex, term: str) -> float:
    tokens = tokenize_text(term)
    if len(tokens) > 1:
            raise Exception("Expecting single token for idf function")
    term = tokens[0]

    doc_count = len(index.docmap)
    term_doc_set = index.index.get(term, set())
    term_doc_count = len(term_doc_set)

    idf_value = math.log((doc_count + 1) / (term_doc_count + 1))
    return idf_value
