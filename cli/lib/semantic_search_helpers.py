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

def verify_embeddings() -> None:
    model = SemanticSearch()
    movies = load_file("movies.json")

    embeddings = model.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str) -> None:
    model = SemanticSearch()

    embedding = model.generate_embedding([query])
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[0][:5]}")
    print(f"Shape: {embedding[0].shape}")

def chunk_text(text: str, size: int, overlap: int) -> list:
    chunks = []
    words = text.split()

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    for i in range(0, len(words), size):
        if i <= overlap:
            chunk_words = words[i:i + size]
        else:
            chunk_words = words[i - overlap:i + size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    
    return chunks