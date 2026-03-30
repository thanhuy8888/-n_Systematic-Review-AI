import re
from rank_bm25 import BM25Okapi
import nltk

class BM25Filter:
    def __init__(self):
        self.corpus = []
        self.tokenized_corpus = []
        self.paper_ids = []
        self.bm25_model = None

    def _tokenize(self, text: str) -> list:
        """Basic whitespace and punctuation tokenizer."""
        # Lowercase and split on non-alphanumeric characters
        if not text:
            return []
        text = text.lower()
        tokens = re.split(r'\W+', text)
        return [t for t in tokens if len(t) > 2] # simple stopword filtering

    def build_index(self, papers: list[dict]):
        """
        Build a BM25 index from a list of paper dictionaries.
        papers should have 'paper_id', 'title', 'abstract'.
        """
        self.corpus = papers
        self.paper_ids = [p['paper_id'] for p in papers]
        
        # Combine title and abstract for text representation
        self.tokenized_corpus = [
            self._tokenize(p.get('title', '') + " " + p.get('abstract', ''))
            for p in papers
        ]
        
        if self.tokenized_corpus:
            self.bm25_model = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 100) -> list[dict]:
        """
        Search the BM25 index and return the top_k papers with scores.
        """
        if not self.bm25_model:
            return []
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25_model.get_scores(tokenized_query)
        
        # Pair up scores with indices and sort
        scored_docs = list(enumerate(doc_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_docs[:top_k]:
            if score <= 0:
                continue # ignore zero scores
                
            paper = self.corpus[idx]
            results.append({
                "paper_id": paper["paper_id"],
                "score": score,
                "title": paper["title"]
            })
            
        return results
