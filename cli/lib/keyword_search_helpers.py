import string 
import os
from nltk.stem import PorterStemmer
from .files import DATA_PATH

# Take text string and process it into search tokens
def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()

    text_tokens = preprocess_text(text).split()
    stopwords = get_stopwords()
    valid = []

    for token in text_tokens:
        if token and (token not in stopwords):
            token = stemmer.stem(token)
            valid.append(token)
    return valid

# Preprocess a string by lowercasing and stripping it of punctuation
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Get list of stopwords from txt file
def get_stopwords() -> list[str]:
    word_file = os.path.join(DATA_PATH, "stopwords.txt")
    with open(word_file, 'r') as f:
        words = f.read().splitlines()
    return words
