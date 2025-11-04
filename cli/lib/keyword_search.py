import string 
import os
from io import StringIO
from .files import load_file, DATA_PATH

DEFAULT_SEARCH_LIMIT = 5

def search_command(query: str) -> list[dict]:
    movie_data = load_file("movies.json")

    results = []

    for movie in movie_data['movies']:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie['title'])
        if match_tokens(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= DEFAULT_SEARCH_LIMIT:
                break

    return results

def match_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def tokenize_text(text: str) -> list[str]:
    text_tokens = preprocess_text(text).split()
    stopwords = get_stopwords()
    valid = []

    for token in text_tokens:
        if token and (token not in stopwords):
            valid.append(token)
    return valid

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def get_stopwords() -> list[str]:
    word_file = os.path.join(DATA_PATH, "stopwords.txt")
    with open(word_file, 'r') as f:
        words = f.read().splitlines()
    return words
