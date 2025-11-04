from .keyword_search import tokenize_text

class InvertedIndex:
    index: dict[str, set]
    docmap: dict[int, object]

    def __init__(self, index, docmap):
        self.index = index
        self.docmap = docmap

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    
    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index[term.lower()])