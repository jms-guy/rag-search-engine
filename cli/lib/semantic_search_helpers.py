from .semantic_search import SemanticSearch

def embed_text(text: str):
    if text == "":
        raise ValueError("text cannot be empty string")
    
    model = SemanticSearch()

    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")