from .semantic_search import SemanticSearch
from .files import load_file

def embed_text(text: str):
    if text == "":
        raise ValueError("text cannot be empty string")
    
    model = SemanticSearch()

    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    model = SemanticSearch()
    movies = load_file("movies.json")

    embeddings = model.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    model = SemanticSearch()

    embedding = model.generate_embedding([query])
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[0][:5]}")
    print(f"Shape: {embedding[0].shape}")